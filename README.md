# Running on Google Colab

1. Zip the whole `Historical_AI` project folder (not just `main/` — it needs the sibling `rag/` and `run/` folders too).
2. Upload `Historical_AI.zip` to the root of your Google Drive (My Drive — no subfolder needed).
3. Open the notebook in Colab.
4. Runtime > Change runtime type > GPU. Whatever GPU you're given is fine (as long as you have 16gb) — even the free-tier T4 is adequate, just a bit slow.
5. Runtime > Run all.

One cell needs you by hand: the Hugging Face login step pops up a token field partway through. Paste a token from an account that's accepted the Gemma license, hit enter, and the rest of the notebook continues on its own.

Everything unzips to `/content/Historical_AI/main` on the Colab machine itself (not Drive) — this keeps things fast, but it's wiped when the runtime disconnects. Zip and copy `saved_conversations/` back to Drive if you want to keep a session's output.

## Notebook

- In this repo: `main/historical_ai_colab.ipynb`
- Shared Colab link: [link](https://colab.research.google.com/drive/1JvnU8IJcgrZwZ9lSf-z-khnbMgQwnYny?usp=drive_link)
