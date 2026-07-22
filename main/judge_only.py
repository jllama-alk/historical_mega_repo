"""Re-run judge_victory + final_debrief over a saved conversation log, without playing the whole game.

Usage: python judge_only.py [path/to/conversations.json] [--force-win]
Judges array 2 and array 3 (the two convince conversations) the same way main.py does,
then runs the same end-of-scenario debrief the real game would print.
"""
import argparse
import json
from langchain_ollama.llms import OllamaLLM

DEFAULT_PATH = "saved_conversations/really-good.json"

SCENARIO = {
    "name": "The First Moroccan Crisis",
    "people_needed": 2,
    "fail_outcome": "The player failed to defuse the First Moroccan Crisis. Historically this is what happens: Germany's attempt to test and break the Anglo-French understanding backfires -- Britain and France read the crisis as proof that Germany will keep probing the alliance system, and their Entente Cordiale hardens from a colonial understanding into a much firmer alliance against Germany."
}


def judge_victory(system_prompt, conversations):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    template = f"""
    You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character.

    You are going to be given the system prompt that was given to the model and the conversation the student had with his partner. You will judge whether the student was able to convince their partner or not.

    System Prompt: {system_prompt}

    The Conversation: {conversations}

    "Really think Hard about the flow of the conversation. Think hard in relating to the conversational partner. A clear, enthusiastic verbal agreement or endorsement from the partner counts as convinced — do not require them to have already finalized or executed the plan, only to have genuinely agreed to it in the conversation."
    """

    template += "\nDid the player succeed in convincing their partner? Only answer with a single word: yes or no."
    return model_orchestrator.invoke(template)


def final_debrief(trial_results, fail_outcome=None):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    trials = "\n\n".join(
        f"Trial {i+1} — {'WON' if t['won'] else 'LOST'}\nSystem Prompt: {t['system_prompt']}"
        for i, t in enumerate(trial_results)
    )

    fail_block = f"\nThe player did not convince enough people, so this is what actually happens: {fail_outcome}\n" if fail_outcome else ""

    template = f"""
    You are a historical analyst. A peace activist in early-1900s Europe attempted to win support from a series of influential figures. Below are the outcomes of those attempts.

    {trials}
    {fail_block}
    The player already knows what happened in their campaign. Do not recap it. Instead, focus entirely on consequences: given who was convinced and who was not, what does this mean for Europe going forward? What political doors opened or closed? What alliances became more or less likely? What risks increased? Be specific to the historical context implied by each character. Write in plain, direct prose — no literary flourishes, no dramatic framing. Two to three short paragraphs.
    """

    return model_orchestrator.invoke(template)


def to_system_and_convo(messages):
    system_prompt = next(m["content"] for m in messages if m["role"] == "system")
    convo = list(zip(
        (m["content"] for m in messages if m["role"] == "user"),
        (m["content"] for m in messages if m["role"] == "assistant"),
    ))
    return system_prompt, convo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH)
    parser.add_argument("--force-win", action="store_true",
                         help="Skip the judge model and treat both convince conversations as convinced")
    args = parser.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    trial_results = []
    for i, messages in enumerate(data[1:3], start=2):
        system_prompt, convo = to_system_and_convo(messages)
        if args.force_win:
            won = True
            print(f"Conversation {i}: CONVINCED (forced)")
        else:
            verdict = judge_victory(system_prompt, convo)
            won = "yes" in verdict.lower()
            print(f"Conversation {i}: {'CONVINCED' if won else 'NOT CONVINCED'} (raw: {verdict.strip()})")
        trial_results.append({"system_prompt": system_prompt, "won": won})

    convinced = sum(t["won"] for t in trial_results)
    fail_outcome = SCENARIO["fail_outcome"] if convinced < SCENARIO["people_needed"] else None
    print(f"\n{SCENARIO['name']} — {convinced}/{SCENARIO['people_needed']} convinced\n")
    print(final_debrief(trial_results, fail_outcome))
