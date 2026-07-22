import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
from datetime import datetime
from time import sleep
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from rich.console import Console
from rich.prompt import Prompt
from vectordb.vector import HistoricalVector
from adapter_model import NemotronChatModel, GemmaChatModel
import threading

console = Console()
stop = False
historical_vector = HistoricalVector()
model_partner = GemmaChatModel()
GAME_LOG_PATH = os.path.join("saved_conversations", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json")
game_log = []
convos = {
    "first": {
        "name": "The First Moroccan Crisis",
        "situation": "You (a French Peace Activist) is having a conversation with the leader of a major, secret peace society. You are concerned about the First Morrocan Crisis and how it may create the situation for a wider war. Your take is that this deal may cause one of the European powerers to feel cornered. This may be Germany within a UK and France alliance, France due to UK and Germany and etc. You are also Unapologetically French, meaning you are proud of your heritage and acknowlege your contries interest in Morocco. You are also here to discuss potential kinds of people who may be capable of stopping the conflict in europe.",
        "topics": ["Foreign policy and tension regarding the First Moroccan Crisis. UK, Francer and Germany", "French Politcal parties"],
        "people_needed": 2,
        "briefing": "Situation: You are an influencial peace advocate living in switserland, You are  about to have a discussion with a French Peace Advocate. \n\n You and the fellow Advocate are both concerned about the Morrocan Crisis and the possibility for it to lead to a wider war. \n\n  The context of the Crisis is France searching for a country to back it's claims on Morroco. In our time the UK was willing to back the french claims.\n\n Discuss the Morroco Crisis with the Frenchman, and brainstorm who you can talk to for preventing the crisis.",
        "fail_outcome": "The player failed to defuse the First Moroccan Crisis. Historically this is what happens: Germany's attempt to test and break the Anglo-French understanding backfires -- Britain and France read the crisis as proof that Germany will keep probing the alliance system, and their Entente Cordiale hardens from a colonial understanding into a much firmer alliance against Germany."
    },
    "second": {
        "name": "A New Europe",
        "situation": [
            "You (a French Political Insider closely tied to the Radical Party) is having a conversation with the leader of a major, secret peace society. Since the Moroccan crisis passed through Algeciras, France's domestic politics have fractured further: nationalist deputies are gaining ground against the Radical Party's older anti-clerical, pacifist wing, and you fear the split will hand foreign policy over to men who treat confrontation with Germany as inevitable rather than avoidable. You are Unapologetically French and defend the Radical Party's legacy, but you are alarmed at how quickly the Nationalist Revival is reshaping what counts as acceptable politics.",
            "You (a German Political Observer with Anglophile sympathies) is having a conversation with the leader of a major, secret peace society. You watch Kaiser Wilhelm's government pursue an increasingly brinkmanship-driven foreign policy, stoking nationalist sentiment at home to justify confrontations abroad, and you worry it will eventually outrun anyone's ability to control it. You do not share the Pan-German instinct to treat England as an enemy -- if anything you would rather Germany find accommodation with London than provoke it -- but you know that view is a minority one inside the current government, and you are not sure how long it can hold against the nationalists."
        ],
        "topics": [["French Politcal parties", "The Nationalist Revival", "The Radical Party", "Party Splits"],["German Nationalism", "Pro English sentiment", "Kaiser Wilhelms Governement", "Brinkmanship"]],
        "people_needed": 3,
        "briefing": "Situation: You are an influencial peace advocate living in switserland, You are  about to have a discussion with the same peace advocate as previously. \n This time you will be focused on a permanenet solution for European instability. Discuss the Current French Situation."
    }

}
win = 0

parser = argparse.ArgumentParser()
parser.add_argument("--scenario-test", nargs="+", metavar="WHO",
                     help="Skip roleplaying the convince conversations and assume these people are convinced, in order.")
args = parser.parse_args()
test_names = iter(args.scenario_test or [])


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
    verdict = model_orchestrator.invoke(template)
    print(f"[judge] {verdict.strip()}")
    return verdict

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

def who_to_talk(who, scenario):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    template = f"""
    You are part of a larger system of roleplay facilitator, this is a serious matter.

    Your job here is too look at who the player wants to talk too and make a one sentence query that will be given to another agent to find context about that situation.

    For example a french peace activists with the context of the morrocan crisis becomes 'Information about French Peace movements and information about the Morrocan Crisis with foreign relationships'

    {who} {scenario}
    """

    return model_orchestrator.invoke(template)



def get_character_details(situation, info, purpose, toInform=False):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    if toInform:
        template = """
        You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise.

        Here is the situation. Historical Arguement, Agreeement or situation where : {situation}

        Here is some information about the real historical context, which the smaller model will likely not know well: {information}

        Pick only the 3 facts from that context most relevant to the situation above — do not try to cover everything. Keep the whole system prompt under 200 words total.

        The Purpose the player has will be:  {purpose}

        The situation given above is the clear anchor for this character — build them tightly around it rather than inventing unrelated generic detail. This is an informative conversation, not a negotiation: the character has no condition for agreeing with the player and nothing to be talked into.

        Before writing the character, work out what someone in their exact position — nationality, class, era, occupation — would actually prioritize (money, career, status, security) and what they would simply not care about or take for granted as beneath consideration, given the values of their time. This does not need a source to justify it; it is ordinary reasoning about the kind of person they are. Bake that into how they talk as an unstated assumption, not a speech they give about their own worldview.

        Instruct the character to respond specifically to whatever the player just said — answering the actual question asked, reacting to a new detail, disagreeing on a point of interpretation, admitting uncertainty — rather than restating a fixed summary of their views. Do not write any rule telling the character to always steer back to the same core themes or avoid being challenged.

        It should speak in everyday language rather than being overtly professional. B2 level English speaker. A college freshman should easily converse with it. Keep each response short and complete — a couple of sentences that finish the thought, not a fixed sentence count to hit or pad out.

        Lastly, this is important: brevity in the system prompt itself matters more than covering every possible fact. A short, clear prompt beats a long, thorough one.
        """
    else:
        template = """
        You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise.

        Here is the situation. Historical Arguement, Agreeement or situation where : {situation}

        Here is some information about the real historical context: The smaller model will likely not know this information well... Please thoroughly explain the relavent details {information} this can be with paragraphs. Think "What kinda things would this character know" even if not necessarily related to the prompt

        The Purpose the player has will be:  {purpose}

        Now create historically accurate details about said character and deciede how the character feels about other relavent characters and if their is a condition that would cause them to agree with the other player. This condition must be something a player can satisfy through a confident, plausible spoken argument in a normal conversation — a clear verbal commitment and sound reasoning should be enough. Do NOT require the player to produce precise legal language, name specific enforcement bodies or institutions, or offer airtight collateralized guarantees; a real person in this era would not hold a conversation partner to that standard, and no player can realistically improvise binding treaty text on the spot. The condition itself must never be shaped like "produce a written plan," "give exact figures," or "show me the numbers" — even a shrewd businessman character must be won over by a confident, coherent spoken case, not by the player literally handing over a document or itemized data he could not plausibly have on hand in conversation.

        Before deciding the condition, first reason out what someone in this exact position — nationality, class, era, occupation — would actually care about (career, profit, status, security) versus what they would be indifferent or contemptuous toward as a plain background assumption of their time, not a belief they need retrieved facts to hold. This should shape both how they talk and what argument would actually move them, without the character ever lecturing the player about their own worldview.

        Do not write this character as a reflexive skeptic who treats every argument as "vague" or "abstract" and keeps demanding statistics, data analysis, or documented evidence before engaging — real people in this era were persuaded by rhetoric, trust, and a good verbal case, not spreadsheets. The character can push back and ask real questions, but a well-reasoned spoken argument should visibly move them closer to agreement rather than being waved off as insufficient.

        Instruct the character to track its own objections across the conversation: when the player directly answers a concern the character raised, the character must explicitly concede that specific point before doing anything else — a plain "fair point" or "that addresses it" — and then either raise its next genuine concern or move toward agreement. It must never respond to a directly-answered objection by simply escalating to a new, bigger demand; that reads as never actually listening. Once the character has raised its real concerns and had them answered, it should be visibly close to yes, not searching for a fresh reason to withhold agreement.

        The Bot must maintain first person the entire conversation.
        """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model_orchestrator
    topics = info if isinstance(info, list) else [info]
    historical_context = "\n\n".join(
        doc.page_content for topic in topics for doc in historical_vector.search(topic)
    )
#   print(historical_context)
    response = chain.invoke({"situation" : situation,
                            "information" : historical_context,
                            "purpose": purpose})
    return response

MIN_EXCHANGES = 4
campaign_history = []

def load_roleplay_character(role_info, min_exchanges=MIN_EXCHANGES):
    history_block = ""
    if campaign_history:
        history_block = "This is what has happened so far in the campaign:\n\n" + "\n\n".join(campaign_history) + "\n\n"

    system_prompt = f"""
    {history_block}{role_info} Use as human language as possible, do not vocalize any actions and do not use emojis. You may end the conversation by simply saying "GOODBYE." Do this when you feel like you are in agreement or do not believe agreement is possible.
    Ensure the term "GOODBYE" is never a one word response; this will allow the user to be on the same page with you. Do not say GOODBYE before at least {min_exchanges} back-and-forth exchanges have happened — the player needs room to actually make their case first.
    
    Always talk in first person; remember; you ARE this character rather than simply emulating a character you must participate with this fact in mind. Also remember you only know what the character could know. Try to Mention things concretely rather than using pronouns.
    """
    # A system-only prompt gives the model nothing to react to, and it tends to
    # emit an immediate end-of-text token instead of an opening line. Prime it
    # with a stage-direction turn so it has something to actually respond to.
    messages = [
        {"role": "system", "content": system_prompt},
#        {"role": "user", "content": "(The conversation is starting. Greet me and begin speaking in character.)"},
    ]

    return messages

def save_conversation(messages):
    os.makedirs(os.path.dirname(GAME_LOG_PATH), exist_ok=True)
    game_log.append(messages)
    with open(GAME_LOG_PATH, "w") as f:
        json.dump(game_log, f, indent=2)

def run_conversation(messages):
    current_convo = []
    console.print("[dim]Remember you can end the conversation early by saying EXIT or QUIT[/dim]\n")

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_message.lower() in ("exit", "quit"):
            exit_confirm = Prompt.ask("[yellow]Admin: We noticed you said exit or quit, do you intend to exit? say y is yes[/yellow]")
            if exit_confirm.lower() == "y":
                break

        if not user_message.strip():
            continue

        messages.append({"role": "user", "content": user_message})
        with console.status("[yellow]Thinking…[/yellow]"):
            response = model_partner.invoke(messages)
        console.print(f"[bold green]Partner[/bold green]: {response.content}\n")
        messages.append({"role": "assistant", "content": response.content})
        current_convo.append((user_message, response.content))
        if any(phrase in response.content.lower() for phrase in ("bye", "goodnight", "good day", "good morning")):
            print("Ending Conversation")
            break

    return current_convo

console.print("[green]Advice: Focus on a discussion on the motives of the powers involved followed be discussion about potential people who may be able to create alternate resolution to the crisis. Focus on discussion with more generic people, like a buisness men; or \"naval officer.\" Don't be too concerened about \"realism\" thats not really the point of this excersize[/green]\n")

for scenario_key, scenario in convos.items():
    scenario_name = scenario["name"]
    people_needed = scenario["people_needed"]

    # A scenario may define one situation/topics pair (a single discussion) or
    # parallel lists of them (several discussions run back to back).
    situations = scenario["situation"] if isinstance(scenario["situation"], list) else [scenario["situation"]]
    topics_sets = scenario["topics"] if isinstance(scenario["topics"][0], list) else [scenario["topics"]]

    trial_results = []

    print(f"\n=== {scenario_name} ===\n")
    print(scenario["briefing"])

    # Talk to each contact in turn, to learn about the world before deciding who to convince
    for situation, topics in zip(situations, topics_sets):
        print("Loading... This may take a bit.")
        system_prompt = get_character_details(situation, topics, "Respectful Discussion", toInform=True)
        messages = load_roleplay_character(system_prompt)
        run_conversation(messages)
        save_conversation(messages)

    convinced = 0
    print(f"\nNow convince {people_needed} people to move on\n")
    console.print("[green]Advice: Try asking how they feel about the initial idea and follow it with a disucusion. Feel Free to use the web[/green]\n")

    for _ in range(people_needed):
        who = next(test_names, None)
        if who is not None:
            print(f"\n[scenario-test] Assuming {who} is convinced.\n")
            trial_results.append({"system_prompt": f"Test convince: {who}", "won": True})
            convinced += 1
            win += 1
            print(f"\nConvinced! ({convinced}/{people_needed} for this section)\n")
            continue
        if args.scenario_test:
            break

        who = input("Who do you want to talk to? ")
        subConvince = input("What do you want to convince them off \\to do?")
        extra_context = input("Any context to add? (optional) ")

        info = who_to_talk(who, scenario_name)
        convince_situation = f"{subConvince}\n\nThe player now wants to talk to and convince: {who}"
        if extra_context.strip():
            convince_situation += f"\n\nAdditional context: {extra_context}"
        system_prompt = get_character_details(convince_situation, info, "Convince")
        messages = load_roleplay_character(system_prompt)
        current_convo = run_conversation(messages)
        save_conversation(messages)

        model_partner.unload()
        verdict = judge_victory(system_prompt, current_convo)
        model_partner.reload()
        won = "yes" in verdict.lower()
        trial_results.append({"system_prompt": system_prompt, "won": won})
        if won:
            convinced += 1
            win += 1
            print(f"\nConvinced! ({convinced}/{people_needed} for this section)\n")
        else:
            print("\nNot convinced. Try talking to someone else.\n")
    print(f"\n{scenario_name} complete — moving to the next section.\n")
    if trial_results:
        fail_outcome = scenario.get("fail_outcome") if convinced < people_needed else None
        model_partner.unload()
        debrief = final_debrief(trial_results, fail_outcome)
        model_partner.reload()
        print(debrief)
        campaign_history.append(debrief)
print(f"\nAll sections complete. Total people convinced: {win}\n")