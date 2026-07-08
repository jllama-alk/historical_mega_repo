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

MODEL_ID   = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_DIR   = "./data"
CSV_PATH   = "./filtered_data.csv"
OUTPUT_DIR = "./nemotronv2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Nemotron's _init_weights does `p /= sqrt(n_layers)` which fails on bitsandbytes
# quantized (byte) tensors. Weights are fully loaded from checkpoint so skip it.
from transformers.modeling_utils import PreTrainedModel as _PTM
_orig_init_missing = _PTM._initialize_missing_keys
_PTM._initialize_missing_keys = lambda self, is_quantized, *a, **kw: (
    None if is_quantized else _orig_init_missing(self, is_quantized, *a, **kw)
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.config.use_cache = False
model.enable_input_require_grads()

# mamba_ssm Triton kernel passes out_proj.weight directly to F.linear, bypassing
# bitsandbytes dispatch. Patch the kernel to dequantize on-the-fly instead of
# dequantizing persistently (saves ~1.1 GB of VRAM vs replacing with BF16 nn.Linear).
import bitsandbytes as _bnb
from mamba_ssm.ops.triton import ssd_combined as _ssd
_orig_mamba_fn = _ssd.mamba_split_conv1d_scan_combined

def _patched_mamba_fn(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
                       initial_states=None, seq_idx=None, dt_limit=(0., float("inf")),
                       return_final_states=False, activation="silu",
                       rmsnorm_weight=None, rmsnorm_eps=1e-6,
                       outproj_weight=None, outproj_bias=None,
                       headdim=None, ngroups=1, norm_before_gate=True):
    if outproj_weight is not None and isinstance(outproj_weight, _bnb.nn.Params4bit):
        outproj_weight = _bnb.functional.dequantize_4bit(
            outproj_weight.data, outproj_weight.quant_state
        ).to(zxbcdt.dtype)
    return _orig_mamba_fn(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
                           initial_states, seq_idx, dt_limit, return_final_states, activation,
                           rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias,
                           headdim, ngroups, norm_before_gate)

# Patch the ssd_combined module AND the model module (which did `from ... import` by name)
_ssd.mamba_split_conv1d_scan_combined = _patched_mamba_fn
_nem_pymod = None
for _, _m in model.named_modules():
    if hasattr(_m, "cuda_kernels_forward"):
        _nem_pymod = sys.modules[type(_m).__module__]
        break
if _nem_pymod:
    _nem_pymod.mamba_split_conv1d_scan_combined = _patched_mamba_fn

# Nemotron's forward does lm_head(hidden).float() → full [B,S,vocab] logit tensor → OOM.
# Patch to use liger fused CE: projects + softmax + CE in chunks, logit tensor never materialised.
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy as _liger_fused_lce
_nem_cls = _nem_pymod.NemotronHForCausalLM
_NemOut  = _nem_pymod.NemotronHCausalLMOutput
_orig_nem_fwd = _nem_cls.forward

def _liger_nem_fwd(self, input_ids=None, labels=None, **kwargs):
    if labels is None:
        return _orig_nem_fwd(self, input_ids=input_ids, labels=None, **kwargs)
    outputs = self.backbone(
        input_ids,
        cache_params=kwargs.get("cache_params"),
        inputs_embeds=kwargs.get("inputs_embeds"),
        output_attentions=kwargs.get("output_attentions"),
        output_hidden_states=kwargs.get("output_hidden_states"),
        return_dict=True,
        use_cache=False,
        cache_position=kwargs.get("cache_position"),
        attention_mask=kwargs.get("attention_mask"),
    )
    hidden = outputs[0]
    shift_h = hidden[:, :-1, :].contiguous()
    shift_l = labels[:, 1:].clone()
    loss = _liger_fused_lce(
        shift_h.view(-1, shift_h.shape[-1]),
        self.lm_head.weight.to(shift_h.dtype),
        shift_l.view(-1).to(shift_h.device),
        ignore_index=-100,
    )
    return _NemOut(loss=loss, logits=None,
                   cache_params=getattr(outputs, "cache_params", None),
                   hidden_states=getattr(outputs, "hidden_states", None))

_nem_cls.forward = _liger_nem_fwd

torch.cuda.empty_cache()

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules="all-linear",
    exclude_modules=["lm_head", "out_proj", "conv1d"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def _tmpl(msgs):
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)

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
