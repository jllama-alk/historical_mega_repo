# Attempts & Failures Log

## Goal

Fine-tune a small LLM to roleplay as historically-grounded characters from the 1900–1914 pre-WWI period — diplomats, intellectuals, colonial figures — using period-accurate knowledge and voice.

---

## Model 1: Gemma-4 2B (`output-g/`, trained ~Jun 25)

**Base:** `unsloth/gemma-4-E2B-it`

### Failures

**OOM on 8GB VRAM during training.**
Gemma-4's forward pass does `logits = logits.float()` before computing cross-entropy. On a `[B, S, vocab]` tensor with vocab ~256k, this doubles memory and kills the backward pass. Fixed by monkey-patching `Gemma4ForConditionalGeneration.forward` to skip the upcast and use `liger_fused_linear_cross_entropy` instead, which chunks the projection + softmax + CE so the full logit tensor is never materialized.

**Per-token entropy metric blew up memory again.**
`SFTTrainer.compute_loss` computes per-token entropy from logits after the loss step — another ~78 MiB allocation that pushed things over. Fixed by subclassing `LeanSFTTrainer` to bypass it and call `Trainer.compute_loss` directly.

**Adapter loading failed at inference.**
`PeftModel.from_pretrained` resolves `target_modules` by name and hits `Gemma4ClippableLinear`, which doesn't match the string patterns. The adapter loads but the weights don't attach. Fixed in `chat.py` by instead calling `get_peft_model` with `"all-linear"` (same as training) and then manually loading the safetensors weights with `load_state_dict(strict=False)`.

**Vision/audio towers wasted ~600 MB of VRAM.**
Gemma-4 is a multimodal model. For text-only training the towers are dead weight but stay loaded. Fixed by deleting `vision_tower`, `audio_tower`, `audio_tower_norm`, `multi_modal_projector` from the inner model after loading.

### Outcome

Training finished (3 epochs, 255 steps). Adapter saved. Inference worked after the loading workaround. Model behavior: acceptable but verbose, with heavy stage-direction formatting even when the system prompt said not to.

---

## Model 2: Ministral-3B Reasoning (`output/`, trained ~Jun 28)

**Base:** `mistralai/Ministral-3-3B-Reasoning-2512`

### Failures

**Model class not registered in transformers.**
`AutoModelForCausalLM.from_pretrained` failed because `Mistral3Config` / `Mistral3ForConditionalGeneration` weren't auto-registered at the version used. Fixed by importing and calling `AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)` before loading.

**Python 3.14 broke datasets fingerprinting.**
`dill._batch_setitems` no longer exists in 3.14, which crashes `datasets`' `Hasher` when fingerprinting the dataset for caching. Fixed by monkey-patching `Hasher.hash` to use `stdlib pickle` (protocol 2) with a `repr()` fallback.

**Same VRAM pressure as Gemma-4.**
The multimodal tower deletion, `LeanSFTTrainer`, and liger patch from the Gemma-4 run were all carried over. The liger patch is now technically a no-op for Ministral (which isn't a multimodal model and doesn't have the same forward), but it doesn't hurt.

### Outcome

Training finished (3 epochs, 255 steps). Adapter saved (~24 MB vs Gemma-4's ~27 MB). Ministral's LoRA target modules were resolved by `"all-linear"` without the loading bug Gemma-4 had, so `from_pretrained` would likely have worked — but `chat.py` still uses the manual load path for consistency.
---

## Model 3: Nemotron-3-Nano-4B (`output-n/`, trained ~Jun 29)

**Base:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`

A 42-layer hybrid Mamba2/Attention/MLP model (21 Mamba + 4 Attention + 17 MLP layers). Inference is in `run/chat_nemotron.py`; the standard `chat.py` is not used for this model.

### Training Failures

**`_initialize_missing_keys` crashed on quantized Mamba tensors.**
Nemotron's `_init_weights` does `p /= sqrt(n_layers)` on every parameter, including bitsandbytes 4-bit tensors which don't support in-place float ops. Fixed by patching `PreTrainedModel._initialize_missing_keys` to no-op when `is_quantized=True` (weights are fully loaded from checkpoint so the init is unnecessary anyway).

**mamba_ssm Triton kernel bypassed bitsandbytes dispatch.**
The `mamba_split_conv1d_scan_combined` Triton kernel passes `out_proj.weight` directly to `F.linear`, bypassing bitsandbytes' `__torch_function__` hook. Fixed by monkey-patching the kernel to dequantize `Params4bit` weights on-the-fly before the call (~1.1 GB VRAM saved vs replacing the layer with a full BF16 linear).

### Inference Failures

**`_pattern_to_list` KeyError on `-` character.**
The model config uses `hybrid_override_pattern` (`"M-M-M-MM-M-M*-..."`) to describe layer types, where `M`=Mamba, `*`=Attention, `-`=MLP. The installed transformers `NemotronHConfig._pattern_to_list` only maps `M/E/*`, missing `-`. Fixed by patching `_pattern_to_list` to add `"-": "mlp"`. Also needed `trust_remote_code=True` since the installed `NemotronHForCausalLM` has no MLP layer support — the remote model code from the HF repo handles all three types correctly.

**MISSING/UNEXPECTED weight keys on load.**
Caused by the above pattern bug producing the wrong layer count (25 instead of 42), so the architecture didn't match the checkpoint. Resolved by the pattern fix.

**`AttributeError: 'Parameter' object has no attribute 'compress_statistics'`**
`PeftModel.from_pretrained` tried to wrap a layer whose `.weight` was a regular `nn.Parameter` (not `Params4bit`) because missing Mamba weights had been randomly re-initialised as unquantized. Root cause was the pattern bug above. Resolved together with it.

**`model.generate()` crashed with `TypeError: 'NoneType' object is not subscriptable` on `cache_position[-1]`.**
Transformers 5.5 pre-initialises `DynamicCache` in `_prepare_cache_for_generation` before calling `prepare_inputs_for_generation`. The remote NemotronH code checks `empty_past_kv = past_key_values is None` and takes a different branch — but the pre-initialised (non-None) `DynamicCache` makes it enter the branch that reads `cache_position[-1]`, which is `None`. Fixed by not using `model.generate()` at all.

**No cache → O(n²) VRAM spikes.**
First working inference used a manual token loop with no cache, so every decode step reprocessed the full growing sequence. Visible as sawtooth VRAM spikes. Fixed by threading `HybridMambaAttentionDynamicCache` through the loop: prefill runs once on the full prompt, then decode passes a single new token + the live cache each step.

**`HybridMambaAttentionDynamicCache` constructor takes `config.hybrid_override_pattern` directly.**
The cache class uses the raw pattern string (not `layers_block_type`) to decide which layers get SSM states. With `trust_remote_code=True` the remote config stores the string as a plain attribute, so this works. With the installed config it would fail (the property recomputes it from `layers_block_type` using a mapping that doesn't include "mlp" → "-").

### Inference Approach

`model.generate()` is not used. `run/chat_nemotron.py` uses a manual prefill + decode loop:
1. Initialise `HybridMambaAttentionDynamicCache` from the loaded remote module via `sys.modules[type(inner_model).__module__]`.
2. Prefill: `model(input_ids=full_prompt, cache_params=cache, use_cache=True, cache_position=arange(seq_len))`.
3. Decode: `model(input_ids=new_token, cache_params=cache, use_cache=True, cache_position=tensor([pos]))` per step.
4. `cache_params` (not `past_key_values`) is threaded through because `NemotronHForCausalLM.forward` takes that kwarg directly; PEFT's `BaseTuner.forward` passes `**kwargs` straight through.

`enable_thinking=False` is passed to `apply_chat_template` to suppress the model's CoT scratchpad.

### Outcome

Training finished (3 epochs, ~255 steps). Inference working with proper O(1)-per-token caching.

---

## Behavioral Failures (both models)

**Stage directions in output.**
Both models produce `**(The diplomat leans forward…)**` blocks by default, apparently because the training data included them or the base models have this habit. System prompts saying "Only respond with your output, do not give descriptors" reduced it but didn't eliminate it in the Gemma-4 runs. The Ministral runs (later conversations) show cleaner output.

**Responses too long.**
The Gemma-4 model in particular runs to 4–6 paragraphs on a simple opening line. Training data may have over-indexed on long "explain your position" exchanges.

**Character doesn't stay under pressure.**
In the Jun 28 conversations, when the French diplomat threatens the character's life, the model answers philosophically rather than staying in the corrupt self-interested persona. The system prompt didn't include enough constraint around how the character handles intimidation.

---

## Data

Training data is split across:
- `data/Single Factual/` — dyadic (2-person) conversations covering period topics (Balkans, Durkheim, Moroccan crises, Edwardian domestic politics, etc.)
- `data/Double/` — group conversations and persuasion scenarios
- `filtered_data.csv` — instruction/input/output format (CSV)
- `data/PDFS/` — source PDFs the JSONL data was presumably generated from

Two JSONL formats are handled: a `messages` format (with optional `system` key) and a `conversations` format (with `role: model` instead of `role: assistant`).

---

## What to Try Next

- Tighten system prompts with explicit behavioral guardrails (no stage directions, short replies, stay in persona under threats).
- Sample more from the `output-g` vs `output` adapter in the same session using `/switch` to compare voice.
- If verbosity persists: reduce `max_new_tokens` in `generate()` or add a length penalty.
- If stage directions persist: filter them from training data and retrain.
