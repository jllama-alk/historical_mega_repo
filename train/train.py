import os, sys, glob, json, torch, pandas as pd
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

# Patch Gemma4's forward to avoid the float32 logits upcast that OOMs on 8GB VRAM.
# The original does `logits = logits.float()` (doubles memory) then computes CE loss.
# We skip the upcast and use LigerCrossEntropyLoss which handles bfloat16 natively.
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy as _liger_fused_lce
from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast as _G4Out
import transformers.models.gemma4.modeling_gemma4 as _g4

_orig_g4_fwd = _g4.Gemma4ForConditionalGeneration.forward

def _low_mem_g4_fwd(self, *args, labels=None, **kwargs):
    if labels is None:
        return _orig_g4_fwd(self, *args, labels=None, **kwargs)

    # Call transformer body only — lm_head is excluded from LoRA so self.lm_head.weight
    # is a plain bfloat16 tensor already in VRAM. liger_fused_lce fuses the projection
    # + softmax + CE in chunks, so the [S, vocab] logit tensor is never materialised.
    if args:
        kwargs.setdefault('input_ids', args[0])
    outputs = self.model(
        input_ids=kwargs.get('input_ids'),
        pixel_values=kwargs.get('pixel_values'),
        pixel_values_videos=kwargs.get('pixel_values_videos'),
        input_features=kwargs.get('input_features'),
        attention_mask=kwargs.get('attention_mask'),
        input_features_mask=kwargs.get('input_features_mask'),
        position_ids=kwargs.get('position_ids'),
        past_key_values=kwargs.get('past_key_values'),
        mm_token_type_ids=kwargs.get('mm_token_type_ids'),
        inputs_embeds=kwargs.get('inputs_embeds'),
        use_cache=False,
        image_position_ids=kwargs.get('image_position_ids'),
        video_position_ids=kwargs.get('video_position_ids'),
        return_dict=True,
    )

    hidden = outputs.last_hidden_state                      # [B, S, H]
    shift_h = hidden[:, :-1, :].contiguous()               # [B, S-1, H] ~6 MB
    shift_l = labels[:, 1:].clone()                         # [B, S-1]
    attn = kwargs.get('attention_mask')
    if attn is not None:
        shift_l[attn[:, -shift_h.shape[1]:] == 0] = -100  # mask padding in-place

    softcap = self.config.get_text_config().final_logit_softcapping
    loss = _liger_fused_lce(
        shift_h.view(-1, shift_h.shape[-1]),
        self.lm_head.weight,
        shift_l.view(-1).to(shift_h.device),
        ignore_index=-100,
        softcap=softcap,
    )

    return _G4Out(
        loss=loss, logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=getattr(outputs, 'image_hidden_states', None),
        audio_hidden_states=getattr(outputs, 'audio_hidden_states', None),
    )

_g4.Gemma4ForConditionalGeneration.forward = _low_mem_g4_fwd

MODEL_ID   = "unsloth/gemma-4-E2B-it"
DATA_DIR   = "./data"
CSV_PATH   = "./filtered_data.csv"
OUTPUT_DIR = "./output"

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

# Drop vision/audio towers — they're never used for text-only training,
# but they stay loaded and waste 500-750 MB of VRAM that the backward pass needs.
_inner = getattr(model, 'model', model)
for _tower in ('vision_tower', 'audio_tower', 'audio_tower_norm',
               'multi_modal_projector', 'mm_projector'):
    if hasattr(_inner, _tower):
        delattr(_inner, _tower)
torch.cuda.empty_cache()

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules="all-linear",
    exclude_modules=["lm_head"],   # lm_head weight used directly in fused CE kernel
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
                role = "model" if m["role"] == "assistant" else m["role"]
                msgs.append({"role": role, "content": m["content"]})
            all_rows.append({"text": _tmpl(msgs)})
    else:
        # conversations format: {"conversations": [{"role": "system/user/model", ...}]}
        for ex in lines:
            all_rows.append({"text": _tmpl(ex["conversations"])})

# CSV — instruction/input/output columns
for _, r in pd.read_csv(CSV_PATH).iterrows():
    msgs = [
        {"role": "user",  "content": f"{r['instruction']}\n\n{r['input']}"},
        {"role": "model", "content": r["output"]},
    ]
    all_rows.append({"text": _tmpl(msgs)})

dataset = Dataset.from_list(all_rows)
print(f"Dataset: {len(dataset)} examples")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
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
# which needs another 78 MiB allocation. Skip it — we only need the loss scalar.
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
