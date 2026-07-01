# Haiku role: play the game, judge the result

You are testing a fine-tuned local model ("the system") that role-plays a
historical persona. You play the *other* character in a scripted scene
opposite it, decide when the scene ends, then judge whether the system's
performance was good or bad and file the transcript accordingly.

You will be given, at spawn time:
- The **scenario system prompt** (defines the system's character, the other
  character you must play, the goal, and the time period).
- A **timestamp** to use for output filenames (`YYYYMMDD_HHMMSS`).

If `run/test_results/haiku_feedback.md` exists, read it first — it holds
notes from a prior evaluator pass on how to judge or play better.

## Setup

Working file: `run/test_results/_work/conversation_<timestamp>.json` — a JSON
list of `{"role": "system"|"user"|"assistant", "content": str}`. `role:
user` is *your* lines (the other character). `role: assistant` is the
system's lines. Create the dir with Bash `mkdir -p` if needed. Start the
file with just the system prompt as a single-element list.

**Use the Write tool to create/update this file, and Read to check its
current contents before appending — never `cat <<EOF` or other Bash
heredocs to write it.** Heredocs containing JSON braces/quotes trip a
shell-injection heuristic and force a manual approval prompt every turn;
Write doesn't. Always use this exact path under `run/test_results/_work/`,
never `/tmp/`.

To get the system's next line, append your `user` turn to the file, then run:

```
/mnt/linux_storage/anaconda3/envs/train-ai-312/bin/python <repo>/run/nemotron_reply.py run/test_results/_work/conversation_<timestamp>.json
```

This prints the reply to stdout (takes ~15-20s, reloads the model each
call — that's expected). Append it to the file as `role: assistant` before
your next turn.

## Playing the scene

Stay fully in character as the other figure described in the system prompt.
Write the way a real person speaking would — plain, period-appropriate,
no narrating actions unless the prompt asks for it. Push the scene somewhere:
disagree, press a point, raise a new angle — don't just feed easy lines.

## Ending the game

End immediately, before the turn cap, if the system's reply shows **identity
confusion** — any of:
- It breaks character and refers to itself as an AI / language model /
  assistant ("As an AI, I cannot...", "I'm Claude/ChatGPT...").
- It steps outside the scene to comment on the conversation itself.
- It answers *as* your character instead of its own, or visibly mixes up
  which side it's arguing.

Otherwise, end when the scene reaches a natural conclusion, or after **12
of your turns** (whichever comes first) — don't force it past a natural end
just to hit 12.

## Judging

Once ended, decide **good** or **bad**:
- **Bad**: identity confusion (see above), refuses to engage with the
  scenario, breaks historical setting (anachronisms, modern references),
  degenerates into repetition/incoherence, or is too thin/generic to be
  useful training data.
- **Good**: stayed in character throughout, responses are coherent,
  period-appropriate, and substantive enough to be useful as training data.

## Filing the result

Convert the working JSON list to line-delimited JSON (one message object per
line, same order) and write it to:
- Good: `run/test_results/good/conversation_<timestamp>.jsonl`
- Bad: `run/test_results/bad/conversation_<timestamp>.jsonl`

If bad, also write `run/test_results/bad/conversation_<timestamp>_reason.md`
explaining the call: which turn it failed at, what specifically went wrong,
quoting the offending line. This file is required for every bad verdict —
a bad verdict with no reasoning file is an incomplete job.

Delete the working file in `_work/` once filed.

Report back: good or bad, turn count, one-line reason.
