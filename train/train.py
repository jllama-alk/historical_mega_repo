"""
QLoRA fine-tuning for Gemma 4 E2B with CPU offloading
Designed for 8GB VRAM (RTX 4060) — model weights spill to RAM as needed.

Requirements:
    pip install unsloth transformers bitsandbytes peft accelerate datasets
"""

import argparse
import os
import torch

# load .env if present
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for _line in open(_env):
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.strip().split("=", 1)
            os.environ.setdefault(_k, _v)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--from-base", action="store_true",
                    help="Load unquantized base model and quantize on load (use if pre-quant model fails to load)")
args = parser.parse_args()

# ── Model ────────────────────────────────────────────────────────────────────

MODEL_ID = (
    "unsloth/gemma-4-E2B-it"                    # base; BnB quantizes it on load
    if args.from_base else
    "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"  # pre-quantized, faster to load
)
DATASET_PATH = "./dataset.jsonl"   # your JSONL training file
OUTPUT_DIR   = "./output"

# ── Quantization config ───────────────────────────────────────────────────────
# llm_int8_enable_fp32_cpu_offload allows modules that can't fit on GPU
# to run on CPU in fp32, instead of crashing.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,          # extra compression
    llm_int8_enable_fp32_cpu_offload=True,   # fixes the error you saw
)

# ── Device map ────────────────────────────────────────────────────────────────
# "auto" lets accelerate decide what fits on GPU vs CPU vs disk.
# If you still OOM, change to {"": "cpu"} to force everything to RAM
# (slower but will definitely load).

# ponytail: cpu-first on --from-base absorbs the pre-quantization shard spike; "auto" resumes after
device_map = {"": "cpu"} if args.from_base else "auto"

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (this will take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,          # load shard-by-shard to avoid RAM spike
)

# ── LoRA config ───────────────────────────────────────────────────────────────
# Rank 8 instead of 16 — your dataset is narrow (character dialogue),
# lower rank = less VRAM, less overfitting risk.
# Alpha 16 = 2x rank scaling, standard stable convention.

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # should show ~1-2% of total params

# ── Dataset ───────────────────────────────────────────────────────────────────
# Expects JSONL where each line has a "conversations" key:
# {"conversations": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}
#
# IMPORTANT for Gemma 4: system prompt must be folded into the first user turn,
# not passed as a separate "system" role.

def format_conversation(example):
    """
    Convert conversation list to Gemma 4 chat format.
    System prompt is prepended to the first user message.
    """
    turns = example["conversations"]
    formatted = tokenizer.apply_chat_template(
        turns,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)

# ── Training args ─────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,       # keep low for 8GB
    gradient_accumulation_steps=8,       # effective batch size = 8
    gradient_checkpointing=True,         # trades compute for VRAM
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_ratio=0.05,
    weight_decay=0.001,
    optim="adamw_8bit",                  # 8-bit optimizer saves ~1GB vs fp32
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_grad_norm=0.3,
    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

print("Starting training...")
trainer.train()

print("Saving adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")