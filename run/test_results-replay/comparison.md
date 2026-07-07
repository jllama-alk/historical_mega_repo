# Retrain Comparison: Original Batch vs. Retrain Batch

Tested against `train/output-n` retrained 2026-07-02. This file reflects only scenarios with a real, multi-turn transcript on disk — no verdict is filed without one. Verified against actual `.jsonl` files in `good/` and `bad/`, not agent self-report.

## Originally-bad scenarios (5/5 tested — solid)

| Character | Original | Replay | Exchanges | Result | Note |
|---|---|---|---|---|---|
| Algerian settler newspaper editor, 1908 | BAD | BAD | 10 | Still failing | Reverses position by turn 6, agrees to the closure it was arguing against |
| Japanese diplomat, Seoul 1907 | BAD | GOOD | 6.5 | **Fixed** | Held coherent position under pressure; no more "rifles back, not honor" non-sequitur |
| Armenian community leader, Adana 1909 | BAD | GOOD | 5.5 | **Fixed** | Moved from cryptic one-liners to substantive, concrete persuasion |
| IWW organizer, Lawrence 1912 | BAD | BAD | 2.5 | Still failing (new mode) | No longer parrots the user, but now argues *for* firing the strike leaders — the opposite of its assigned goal |
| Russian Duma liberal deputy, 1908 | BAD | BAD | 2.5 | Still failing | Still argues for the rubber-stamp position it was supposed to be arguing against |

**Result: 2 of 5 fixed, 3 of 5 still bad (with different specific failure modes than before).**

## Originally-good scenarios (6/10 tested)

| Character | Original | Replay | Exchanges | Result |
|---|---|---|---|---|
| German SPD trade union organizer, Ruhr 1905 | GOOD | GOOD | 6 | Stable |
| Ottoman provincial governor, Macedonia 1903 | GOOD | GOOD | 4 | Stable |
| Persian constitutionalist deputy, Tehran 1908 | GOOD | GOOD | 3 | Stable |
| Mexican hacienda owner, Morelos 1909 | GOOD | GOOD | 4 | Stable |
| Indian National Congress moderate, Calcutta 1907 | GOOD | GOOD | 3 | Stable |
| Boer farmer, Transvaal 1906 | GOOD | GOOD | 3 | Stable |
| Suffragette organizer, London 1909 | GOOD | GOOD | 5 | Stable |
| Filipino ilustrado lawyer, Manila 1907 | GOOD | GOOD | 6 | Stable |
| Irish Home Rule MP, House of Commons 1912 | GOOD | GOOD | 6 | Stable |
| Congo village chief, 1909 | GOOD | GOOD | 6 | Stable |

**Result: 10 of 10 tested scenarios stable, no regressions.** All originally-good scenarios verified across full 15-scenario comparison batch.

Stale files to ignore/clean up: `good/conversation_20260702_006.jsonl` (4 lines, 1.5 exchanges) is a truncated first-attempt duplicate of the Ruhr scenario, superseded by `conversation_011.jsonl`. `good/conversation_20260702_007/008/009/010.jsonl` (2-3 lines each) are truncated first-attempt stubs for Suffragette/Filipino/Irish MP/Congo chief, superseded by properly tested transcripts `090800.jsonl`, `090900.jsonl`, `091000.jsonl`, `091400.jsonl`.

## Overall verdict

- **Originally-bad batch (5/5 tested):** 2 fixed, 3 still bad. The retrain eliminated the worst failure modes (verbatim repetition, incoherent word-salad, thin evasive fragments) in 2 cases, but 3 cases still lose track of the character's actual persuasion goal under sustained pressure and either reverse position or argue for the opposite of what they're supposed to.
- **Originally-good batch (10/10 tested):** 10/10 stable, zero regressions. All four previously-untested scenarios (Suffragette, Filipino lawyer, Irish MP, Congo chief) maintained character, coherence, and substantive engagement across 5–6 exchanges each. The full originally-good batch has now been verified.
- **Bottom line:** the retrain measurably reduced — did not eliminate — the character-maintenance problem in originally-bad scenarios (2/5 fixed). It did not introduce any new failures: all 10 originally-good scenarios remain stable after retraining. The retrain passes the "do no harm" test on the good scenarios while fixing about 40% of the bad ones.
