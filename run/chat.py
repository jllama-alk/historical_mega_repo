#!/usr/bin/env python3
import json, sys, datetime, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
from rich.console import Console
from rich.prompt import Prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel as _PTM
from peft import PeftModel

# Nemotron-H config: installed transformers doesn't map '-' (MLP layer) in _pattern_to_list
# trust_remote_code=True uses NVIDIA's remote config which handles this natively;
# this patch is a fallback in case the local NemotronHConfig is invoked first
try:
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    @staticmethod
    def _fixed_pattern_to_list(pattern: str) -> list:
        mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [mapping[c] for c in pattern]
    NemotronHConfig._pattern_to_list = _fixed_pattern_to_list
except ImportError:
    pass

# Mistral3 registration (no-op for non-Mistral models)
try:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
    AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
except Exception:
    pass

# Nemotron: _init_weights does p /= sqrt(n_layers) which errors on bnb quantized tensors
_orig_init_missing = _PTM._initialize_missing_keys
_PTM._initialize_missing_keys = lambda self, is_quantized, *a, **kw: (
    None if is_quantized else _orig_init_missing(self, is_quantized, *a, **kw)
)

# Nemotron: mamba_ssm Triton kernel passes out_proj.weight directly to F.linear
try:
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
                outproj_weight, outproj_weight.quant_state
            ).to(torch.bfloat16)
        return _orig_mamba_fn(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
            initial_states=initial_states, seq_idx=seq_idx, dt_limit=dt_limit,
            return_final_states=return_final_states, activation=activation,
            rmsnorm_weight=rmsnorm_weight, rmsnorm_eps=rmsnorm_eps,
            outproj_weight=outproj_weight, outproj_bias=outproj_bias,
            headdim=headdim, ngroups=ngroups, norm_before_gate=norm_before_gate,
        )
    _ssd.mamba_split_conv1d_scan_combined = _patched_mamba_fn
except ImportError:
    pass

ADAPTER = Path(__file__).parent.parent / "train" / "output-n"
SAVE_DIR = Path(__file__).parent

console = Console()


def load_model(adapter_path: Path):
    base = json.loads((adapter_path / "adapter_config.json").read_text())["base_model_name_or_path"]
    console.print(f"[yellow]Loading {base}…[/yellow]")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def swap_adapter(model, adapter_path: Path):
    model.load_adapter(adapter_path, adapter_name="default")


def generate(tokenizer, model, messages, max_new_tokens=512):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    adapter_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ADAPTER
    tokenizer, model = load_model(adapter_path)

    console.print("\n[bold]System prompt[/bold] (blank line to finish, empty to skip):")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    system = "\n".join(lines).strip()

    messages = [{"role": "system", "content": system}] if system else []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"conversation_{ts}.jsonl"
    console.print(f"[green]Saving to {save_path.name}[/green]")
    console.print("[dim]'quit' to exit · '/switch <path>' to hot-swap adapter[/dim]\n")

    with open(save_path, "w") as f:
        if system:
            f.write(json.dumps({"role": "system", "content": system}) + "\n")

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.startswith("/switch "):
                new_path = Path(user_input[8:].strip())
                with console.status(f"[yellow]Swapping to {new_path.name}…[/yellow]"):
                    swap_adapter(model, new_path)
                console.print(f"[green]Swapped to {new_path.name}[/green]\n")
                continue

            messages.append({"role": "user", "content": user_input})
            f.write(json.dumps({"role": "user", "content": user_input}) + "\n")
            f.flush()

            with console.status("[yellow]Thinking…[/yellow]"):
                reply = generate(tokenizer, model, messages)

            messages.append({"role": "assistant", "content": reply})
            f.write(json.dumps({"role": "assistant", "content": reply}) + "\n")
            f.flush()

            console.print(f"[bold green]AI[/bold green]: {reply}\n")

    console.print(f"\n[dim]Saved to {save_path}[/dim]")


if __name__ == "__main__":
    main()
