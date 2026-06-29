#!/usr/bin/env python3
"""
Inference script for the Nemotron-H LoRA adapter.

model.generate() is intentionally not used: transformers 5.5 pre-initialises
DynamicCache before calling prepare_inputs_for_generation, but the remote
NemotronH code expects past_key_values=None so it can create its own
HybridMambaAttentionDynamicCache. This collision causes a crash on
cache_position[-1]. The manual sampling loop below sidesteps all of that.

Trade-off: O(n^2) compute — every new token re-processes the full sequence
from scratch because the SSM state is not properly threaded back. For the
token counts typical in this chat app (~few hundred tokens per turn) this
is acceptable.
"""
import json, sys, datetime, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import torch
from rich.console import Console
from rich.prompt import Prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel as _PTM
from peft import PeftModel

ADAPTER = Path(__file__).parent.parent / "train" / "output-n"
SAVE_DIR = Path(__file__).parent

console = Console()

# ── patches ──────────────────────────────────────────────────────────────────

# Nemotron-H config: installed transformers maps only M/E/* in _pattern_to_list,
# missing '-' which the remote config uses for MLP layers (17 of 42 layers).
# trust_remote_code=True uses NVIDIA's remote config that handles this natively;
# this patch is a belt-and-suspenders fallback in case the local class is invoked.
try:
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    @staticmethod
    def _fixed_pattern_to_list(pattern: str) -> list:
        mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [mapping[c] for c in pattern]
    NemotronHConfig._pattern_to_list = _fixed_pattern_to_list
except ImportError:
    pass

# Nemotron: _init_weights does p /= sqrt(n_layers) which errors on bnb 4-bit tensors.
_orig_init_missing = _PTM._initialize_missing_keys
_PTM._initialize_missing_keys = lambda self, is_quantized, *a, **kw: (
    None if is_quantized else _orig_init_missing(self, is_quantized, *a, **kw)
)

# Nemotron: mamba_ssm Triton kernel passes out_proj.weight directly to F.linear,
# bypassing bitsandbytes dispatch. Dequantize on-the-fly instead.
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

# ── model loading ─────────────────────────────────────────────────────────────

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


# ── generation ────────────────────────────────────────────────────────────────

def generate(tokenizer, model, messages, max_new_tokens=512, temperature=0.7):
    """
    Manual token-by-token sampling loop. Avoids model.generate() which in
    transformers 5.5 pre-initialises DynamicCache, colliding with the remote
    NemotronH code that expects past_key_values=None on the first call.
    enable_thinking=False skips the model's internal CoT scratchpad.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    eos = tokenizer.eos_token_id

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids, return_dict=True)
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            if next_tok.item() == eos:
                break

    prompt_len = tokenizer(text, return_tensors="pt").input_ids.shape[1]
    return tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)


# ── chat loop ─────────────────────────────────────────────────────────────────

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
    console.print("[dim]'quit' to exit[/dim]\n")

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
