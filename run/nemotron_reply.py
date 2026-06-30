#!/mnt/linux_storage/anaconda3/envs/train-ai-312/bin/python
"""One-shot Nemotron inference for the test harness.

Run with the train-ai-312 conda env (has the matching transformers/mamba_ssm/
bitsandbytes versions chat_nemotron.py's patches target):
  /mnt/linux_storage/anaconda3/envs/train-ai-312/bin/python run/nemotron_reply.py conversation.json
or just `./nemotron_reply.py conversation.json` (executable, shebang points there).

conversation.json: JSON list of {"role": "system"|"user"|"assistant", "content": str}
Prints the model's next reply to stdout. Nothing else goes to stdout.

ponytail: reloads the model fresh every call (~10-20s) instead of staying
resident. Simplest thing that works for a 12-turn test game; if turn latency
becomes the bottleneck, add a --serve mode that loads once and answers
requests over a local socket.
"""
import sys, json
from pathlib import Path

def main():
    messages = json.loads(Path(sys.argv[1]).read_text())

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # keep model-loading chatter off stdout
    import chat_nemotron as cn
    tokenizer, model = cn.load_model(cn.ADAPTER)
    reply = cn.generate(tokenizer, model, messages)
    sys.stdout = real_stdout

    print(reply)

if __name__ == "__main__":
    main()
