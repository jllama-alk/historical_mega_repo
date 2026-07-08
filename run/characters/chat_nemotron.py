#!/usr/bin/env python3
"""Interactive CLI chat loop for the Nemotron-H LoRA adapter."""
import json, sys, datetime, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from nemotron_inference import ADAPTER, load_model, generate

SAVE_DIR = Path(__file__).parent

console = Console()


# ── chat loop ─────────────────────────────────────────────────────────────────

def main(adapter_path: Path = ADAPTER):
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
    main(Path(sys.argv[1])) if len(sys.argv) > 1 else main()
