import os, sys, glob, json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch, pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers import Trainer as _Trainer
from peft import LoraConfig, get_peft_model, TaskType

_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for _line in open(_env):
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)

# Python 3.14 broke dill's _batch_setitems — patch datasets hashing to use stdlib pickle
if sys.version_info >= (3, 14):
    import hashlib as _hashlib, pickle as _stdlib_pickle
    from datasets import fingerprint as _fp_module
    @classmethod
    def _safe_hash(cls, value):
        try: return _hashlib.md5(_stdlib_pickle.dumps(value, protocol=2)).hexdigest()
        except Exception: return _hashlib.md5(repr(value).encode()).hexdigest()
    _fp_module.Hasher.hash = _safe_hash

MODEL_ID   = "google/gemma-4-E2B-it"
DATA_DIR   = "./data"
CSV_PATH   = "./filtered_data.csv"
OUTPUT_DIR = "./output-gemmav2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)

model.config.use_cache = False
model.enable_input_require_grads()

# Gemma-4's 262K vocab makes the standard [B,S,vocab] logits tensor huge enough
# to OOM on its own during cross-entropy (16GB+ at seq 8192). Liger's fused
# linear CE computes the loss without ever materializing full logits.
#
# This checkpoint loads as Gemma4ForConditionalGeneration (the multimodal
# wrapper - AutoModelForCausalLM resolves to it regardless, since that's what
# the checkpoint's "architectures" field declares), not the plain
# Gemma4ForCausalLM Liger's helper expects. Its .model.language_model is the
# actual Gemma4TextModel, so patch RMSNorm/GEGLU there, and hand-roll the
# fused-CE forward on the outer class since Liger has no wrapper for it.
from types import MethodType
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4_text
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result
from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast

apply_liger_kernel_to_gemma4_text(model=model.model.language_model, fused_linear_cross_entropy=False)

def _liger_gemma4_forward(self, input_ids=None, labels=None, **kwargs):
    kwargs.pop("logits_to_keep", None)
    kwargs.pop("return_dict", None)
    kwargs["use_cache"] = False
    outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True, **kwargs)
    text_config = self.config.get_text_config()
    result = LigerForCausalLMLoss(
        hidden_states=outputs.last_hidden_state,
        lm_head_weight=self.lm_head.weight,
        labels=labels,
        hidden_size=text_config.hidden_size,
        final_logit_softcapping=getattr(text_config, "final_logit_softcapping", None),
    )
    loss, *_ = unpack_cross_entropy_result(result)
    return Gemma4CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values)

model.forward = MethodType(_liger_gemma4_forward, model)

torch.cuda.empty_cache()

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules="all-linear",
    exclude_modules=["lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def _tmpl(msgs):
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

all_rows = []

# JSONL files — two formats detected by presence of top-level "system" key
for path in sorted(glob.glob(f"{DATA_DIR}/**/*.jsonl", recursive=True)):
    with open(path) as f:
        first_line = f.readline()
    first = json.loads(first_line)
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if "messages" in first:
        # converted format: {"system": "...", "messages": [{"role": "user/assistant", ...}]}
        for ex in lines:
            msgs = ([{"role": "system", "content": ex["system"]}] if ex.get("system") else [])
            for m in ex["messages"]:
                msgs.append({"role": m["role"], "content": m["content"]})
            all_rows.append({"text": _tmpl(msgs)})
    else:
        # conversations format: {"conversations": [{"role": "system/user/model", ...}]}
        for ex in lines:
            msgs = [{"role": "assistant" if m["role"] == "model" else m["role"], "content": m["content"]}
                    for m in ex["conversations"]]
            all_rows.append({"text": _tmpl(msgs)})

# CSV — instruction/input/output columns
for _, r in pd.read_csv(CSV_PATH).iterrows():
    msgs = [
        {"role": "user",  "content": f"{r['instruction']}\n\n{r['input']}"},
        {"role": "assistant", "content": r["output"]},
    ]
    all_rows.append({"text": _tmpl(msgs)})

dataset = Dataset.from_list(all_rows)
print(f"Dataset: {len(dataset)} examples")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=10,
    weight_decay=0.001,
    optim="adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_grad_norm=0.3,
    report_to="none",
    dataset_text_field="text",
    max_length=2048,
)

# SFTTrainer normally computes per-token entropy from logits after the loss step,
# which needs another allocation. Skip it — we only need the loss scalar.
class LeanSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["use_cache"] = False
        loss, outputs = _Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        return (loss, outputs) if return_outputs else loss

trainer = LeanSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

torch.cuda.empty_cache()
print("Starting training...")
trainer.train()

print("Saving adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")
