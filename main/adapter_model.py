"""LangChain-compatible wrapper around the Nemotron-H LoRA adapter, for use in RAG chains."""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "run" / "characters"))
from nemotron_inference import ADAPTER, load_model, generate


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

def _to_role_dicts(messages: list[BaseMessage]) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role_map.get(m.type, m.type), "content": m.content} for m in messages]


class NemotronChatModel(BaseChatModel):
    adapter_path: Path = ADAPTER
    max_new_tokens: int = 624
    temperature: float = 0.7

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._tokenizer, self._model = load_model(self.adapter_path)

    @property
    def _llm_type(self) -> str:
        return "nemotron-h-lora"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        reply = generate(
            self._tokenizer, self._model, _to_role_dicts(messages),
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

def load_inference_model(adapter_path):
    # Load the base model exactly as in training (no Liger — train_gemma_plain.py
    # doesn't use it either, so the architectures match without any patching).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


class GemmaChatModel(BaseChatModel):
    """Same model construction + pipeline() call verified working in diag3.py,
    just wrapped as a BaseChatModel instead of going through HuggingFacePipeline.
    """
    base_model_id: str = "google/gemma-4-E2B-it"
    adapter_path: str = "/mnt/linux_storage/projects/Historical_AI/train/output-gemmav2-plain"
    max_new_tokens: int = 200
    temperature: float = 0.3

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        model = load_inference_model(self.adapter_path)
        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )

    @property
    def _llm_type(self) -> str:
        return "gemma-qlora"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        prompt = self._tokenizer.apply_chat_template(
            _to_role_dicts(messages), tokenize=False, add_generation_prompt=True,
        )
        reply = self._pipe(prompt)[0]["generated_text"][len(prompt):]
        # Gemma4's chat template uses "<turn|>" as a literal closing marker rather than
        # a registered special token, so skip_special_tokens doesn't strip it — cut it off.
        reply = reply.split("<turn|>")[0].strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


if __name__ == "__main__":
    from langchain_core.messages import SystemMessage, HumanMessage
    model = GemmaChatModel()
    messages = [
        SystemMessage(content="You (a French Peace Activist) is having a conversation with the leader of a major, secret peace society. You are concerned about the First Morrocan Crisis and how it may create the situation for a wider war. Your take is that this deal may cause one of the European powerers to feel cornered. This may be Germany within a UK and France alliance, France due to UK and Germany and etc"),
        HumanMessage(content="Hello thank you for this oppurtunity to have this conversation?"),
    ]
    print(model.invoke(messages).content)