import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_gemma4_text

base_model_id = "google/gemma-4-E2B-it"
adapter_path = "/mnt/linux_storage/projects/Historical_AI/train/output-gemmav2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
)
apply_liger_kernel_to_gemma4_text(model=base_model.model.language_model, fused_linear_cross_entropy=False)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

msgs = [{"role": "user", "content": "This is a test, just say hello, Gemma here and nothing else!"}]
prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=40, temperature=0.3)

print("=== raw pipeline object, called directly ===")
print(pipe(prompt)[0]["generated_text"][len(prompt):])

print()
print("=== same pipe, wrapped in HuggingFacePipeline.invoke() like adapter_model.py ===")
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)
print(llm.invoke(prompt))
