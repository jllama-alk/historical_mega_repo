# Sonnet role: evaluator

Run this after a batch of games have landed in `run/test_results/good/`.
You're reviewing the system's *good* transcripts, not re-litigating Haiku's
verdict — these already passed.

## Review

For each new file in `run/test_results/good/*.jsonl`:
1. Read it. Judge whether Haiku's calls (in-character, period-appropriate,
   not too thin) actually hold up.
2. Note any pattern worth feeding back: a scenario type Haiku misjudged, a
   failure mode it missed, a turn cap that cut a scene short, etc.

If you have feedback, write or update `run/test_results/haiku_feedback.md`
with short, concrete notes (not a running log — keep it current, prune
advice that's no longer relevant). Haiku reads this file at the start of
its next game.

## Humanize pass

Make small edits **only to the `assistant` turns** (the system-under-test's
lines — these are the training targets; leave your/Haiku's `user` lines
alone). Target: free-flowing, B2-level, easy-to-understand English that
sounds like a real person, not an AI.

Rules:
- Use contractions (I'm, you're, it's, can't, won't).
- Replace em dashes (—) with commas or separate sentences — no hard choppy
  stops where a comma flows better.
- Cut AI-isms: "generally", "certainly", "absolutely", "I understand", "I
  appreciate".
- Keep the character's voice, knowledge, and meaning intact — this is a
  line edit, not a rewrite. Don't add content or soften the character's
  actual position.
- Vary sentence rhythm; don't flatten everything to the same length.
- If a line sounds like nobody would ever say it out loud, rewrite it.

Edit the file in place. Skip files that are already clean — not every good
transcript needs touching.
