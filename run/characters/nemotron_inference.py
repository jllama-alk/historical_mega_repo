"""
Model loading + generation for the Nemotron-H LoRA adapter.

model.generate() is intentionally not used: transformers 5.5 pre-initialises
DynamicCache before calling prepare_inputs_for_generation, but the remote
NemotronH code expects past_key_values=None so it can create its own
HybridMambaAttentionDynamicCache. This collision causes a crash on
cache_position[-1]. The manual sampling loop below sidesteps all of that.

Cache is properly threaded: prefill runs once on the full prompt, then each
decode step passes only the single new token + the live HybridMambaAttentionDynamicCache,
giving O(1) per token after the prefill instead of O(n^2).
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel as _PTM
from peft import PeftModel

ADAPTER = Path(__file__).parent.parent.parent / "train" / "output-n"

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
# Scoped to Nemotron only — blanket-skipping this for every quantized model corrupts
# weight init on other architectures (e.g. silently produced garbage output on Gemma4).
_orig_init_missing = _PTM._initialize_missing_keys
def _patched_init_missing(self, is_quantized, *a, **kw):
    if is_quantized and "nemotron" in type(self).__module__:
        return None
    return _orig_init_missing(self, is_quantized, *a, **kw)
_PTM._initialize_missing_keys = _patched_init_missing

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
    import json
    base = json.loads((adapter_path / "adapter_config.json").read_text())["base_model_name_or_path"]

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
    Prefill + cached decode loop.
    - Prefill: one forward pass on the full prompt, initialises HybridMambaAttentionDynamicCache
    - Decode: one token at a time, passing only the new token + live cache each step
    cache_params is passed directly to NemotronHForCausalLM.forward (not past_key_values)
    via BaseTuner.forward → NemotronHForCausalLM.forward(**kwargs).
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    eos = tokenizer.eos_token_id

    # HybridMambaAttentionDynamicCache lives in the remote module loaded by trust_remote_code
    inner = model.base_model.model  # NemotronHForCausalLM
    remote_mod = sys.modules[type(inner).__module__]
    HybridCache = remote_mod.HybridMambaAttentionDynamicCache
    # Remote code bug: update_conv_state / update_ssm_state call .device on the list instead of the element
    def _update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        dev = self.conv_states[layer_idx].device
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(dev)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(dev)
        return self.conv_states[layer_idx]

    def _update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    HybridCache.update_conv_state = _update_conv_state
    HybridCache.update_ssm_state = _update_ssm_state

    # Remote bug: NemotronHBlock.forward calls attention mixer without past_key_value=cache_params,
    # so the KV cache is never populated/read during decode — attention sees only the single new token.
    NemotronHBlock = remote_mod.NemotronHBlock
    def _block_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
            residual = hidden_states
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if self.block_type == "mamba":
                hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position)
            elif self.block_type == "attention":
                hidden_states = self.mixer(
                    hidden_states, attention_mask=attention_mask,
                    past_key_value=cache_params, cache_position=cache_position,
                )[0]
            elif self.block_type == "mlp":
                hidden_states = self.mixer(hidden_states)
            else:
                raise ValueError(f"Invalid block_type: {self.block_type}")
            hidden_states = residual + hidden_states
            return hidden_states
    NemotronHBlock.forward = _block_forward

    cache = HybridCache(inner.config, batch_size=1, dtype=torch.bfloat16, device=model.device)
    cache.conv_kernel_size = inner.config.conv_kernel  # cuda_kernels_forward reads this but __init__ never sets it

    generated = []
    with torch.inference_mode():
        # Prefill
        cache_pos = torch.arange(seq_len, device=model.device)
        out = model(input_ids=input_ids, cache_params=cache, use_cache=True,
                    cache_position=cache_pos, return_dict=True)
        cache = out.cache_params

        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated.append(next_tok.item())

        # Decode: single token per step
        for step in range(max_new_tokens - 1):
            if next_tok.item() == eos:
                break
            cache_pos = torch.tensor([seq_len + step], device=model.device)
            out = model(input_ids=next_tok, cache_params=cache, use_cache=True,
                        cache_position=cache_pos, return_dict=True)
            cache = out.cache_params
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok.item())

    if generated and generated[-1] == eos:
        generated = generated[:-1]
    return tokenizer.decode(generated, skip_special_tokens=True)
