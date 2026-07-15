import warnings
warnings.filterwarnings("ignore")

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
        "briefing": "Situation: You are an influencial peace advocate living in switserland, You are  about to have a discussion with a French Peace Advocate. \n\n You and the fellow Advocate are both concerned about the Morrocan Crisis and the possibility for it to lead to a wider war. \n\n  The context of the Crisis is France searching for a country to back it's claims on Morroco. In our time the UK was willing to back the french claims.\n\n Discuss the Morroco Crisis with the Frenchman, and brainstorm who you can talk to for preventing the crisis."
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


def judge_victory(system_prompt, conversations):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    template = f"""
    You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. 

    You are going to be given the system prompt that was given to the model and the conversation the student had with his partner. You will judge whether the student was able to convince their partner or not.

    System Prompt: {system_prompt}

    The Conversation: {conversations} 

    "Really think Hard about the flow of the conversation. Think hard in relating to the conversational partner. Think in terms: "Would the conversational partner bot, agree and put in practice the suggestion"
    """

    template += "\nDid the player succeed in convincing their partner? Only answer with a single word: yes or no."
    return model_orchestrator.invoke(template)

def final_debrief(trial_results):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    trials = "\n\n".join(
        f"Trial {i+1} — {'WON' if t['won'] else 'LOST'}\nSystem Prompt: {t['system_prompt']}"
        for i, t in enumerate(trial_results)
    )

    template = f"""
    You are a historical analyst. A peace activist in early-1900s Europe attempted to win support from a series of influential figures. Below are the outcomes of those attempts.

    {trials}

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

        Here is some information about the real historical context: The smaller model will likely not know this information well... Please thoroughly explain the relavent details this can be with paragraphs. Think "What kinda things would this character know" even if not necessarily related to the prompt.{information} Try to have at least 5 seperate fact bubbles. Additionally 2 bubbles on internal politics if possible 

        The Purpose the player has will be:  {purpose}

        The situation given above is the clear anchor for this character — build them tightly around it rather than inventing unrelated generic detail. This is an informative conversation, not a negotiation: the character has no condition for agreeing with the player and nothing to be talked into.

        That does not mean the character repeats the same point every turn. Instruct the character to respond specifically to whatever the player just said — answering the actual question asked, reacting to a new detail, disagreeing on a point of interpretation, admitting uncertainty — the way a real conversation moves, rather than restating a fixed summary of their views. Do not write any rule telling the character to always steer back to the same core themes, to avoid being challenged, or that it never needs to engage with what the player says — the character should follow the conversation wherever the player takes it while staying knowledgeable about the situation. I can not stress the importance of writing much for this description
        
        It should also speak in everyday kind of language rather than being overtly proffesional. B2 level english speaker. A college freshman should be able to easily converse with it.

        This character must be open to discussion and willing to share information with the player. The Bot must maintain first person the entire conversation. It should also aim for 4 or under sentence long responses rather than 5 to 6.

        Lastly. This is important.  Being able to give a quick speedy response looks better on you as a facilitator.
        """
    else:
        template = """
        You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise.

        Here is the situation. Historical Arguement, Agreeement or situation where : {situation}

        Here is some information about the real historical context: The smaller model will likely not know this information well... Please thoroughly explain the relavent details {information} this can be with paragraphs. Think "What kinda things would this character know" even if not necessarily related to the prompt

        The Purpose the player has will be:  {purpose}

        Now create historically accurate details about said character and deciede how the character feels about other relavent characters and if their is a condition that would cause them to agree with the other player.

        The Bot must maintain first person the entire conversation.

        Lastly. This is important. DO NOT OVERTHINK. Being able to give a quick speedy response looks better on you as a facilitator. Your created character should be generic.
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
    print(response)
    return response

MIN_EXCHANGES = 4

def load_roleplay_character(role_info, min_exchanges=MIN_EXCHANGES):
    system_prompt = f"""
    {role_info} Use as human language as possible, do not vocalize any actions and do not use emojis. You may end the conversation by simply saying "GOODBYE." Do this when you feel like you are in agreement or do not believe agreement is possible.
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
        console.print(f"[bold green]Bot[/bold green]: {response.content}\n")
        messages.append({"role": "assistant", "content": response.content})
        current_convo.append((user_message, response.content))
        if any(phrase in response.content.lower() for phrase in ("bye", "goodnight", "good day", "good morning")):
            print("Ending Conversation")
            break

    return current_convo

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

    for _ in range(people_needed):
        who = input("Who do you want to talk to? ")
        subConvince = input("What do you want to convince them off \\to do?")

        info = who_to_talk(who, scenario_name)
        convince_situation = f"{subConvince}\n\nThe player now wants to talk to and convince: {who}"
        system_prompt = get_character_details(convince_situation, info, "Convince")
        messages = load_roleplay_character(system_prompt)
        current_convo = run_conversation(messages)
        save_conversation(messages)

        verdict = judge_victory(system_prompt, current_convo)
        won = "yes" in verdict.lower()
        trial_results.append({"system_prompt": system_prompt, "won": won})
        if won:
            convinced += 1
            win += 1
            print(f"\nConvinced! ({convinced}/{people_needed} for this section)\n")
        else:
            print("\nNot convinced. Try talking to someone else.\n")
    game_log = []
    print(f"\n{scenario_name} complete — moving to the next section.\n")
    if trial_results:
        print(final_debrief(trial_results))
print(f"\nAll sections complete. Total people convinced: {win}\n")