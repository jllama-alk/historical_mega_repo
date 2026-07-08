import warnings
warnings.filterwarnings("ignore")

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
convos = [["You (a French Peace Activist) is having a conversation with the leader of a major, secret peace society. You are concerned about the First Morrocan Crisis and how it may create the situation for a wider war. Your take is that this deal may cause one of the European powerers to feel cornered. This may be Germany within a UK and France alliance, France due to UK and Germany and etc", "The First Moroccan Crisis", 2]]
win = 0


def judge_victory(system_prompt, conversations):
    model_orchestrator = OllamaLLM(model="gemma4:e4b", keep_alive=0)

    template = f"""
    You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. 

    You are going to be given the system prompt that was given to the model and the conversation the student had with his partner. You will judge whether the student was able to convince their partner or not.

    System Prompt: {system_prompt}

    The Conversation: {conversations} 
    """

    template += "\nDid the player succeed in convincing their partner? Only answer with a single word: yes or no."
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

        Here is some information about the real historical context: {information}

        The Purpose the player has will be:  {purpose}

        The situation given above is the clear main focus of this character — build them tightly around it rather than inventing unrelated generic detail. This is an informative conversation, not a debate or negotiation, so do not give the character a condition for agreeing with the player or frame them as needing to be convinced of anything.
        
        This character must be open to discussion and willing to share information with the player. The Bot must maintain first person the entire conversation.

        Lastly. This is important. DO NOT OVERTHINK. Being able to give a quick speedy response looks better on you as a facilitator.
        """
    else:
        template = """
        You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise.

        Here is the situation. Historical Arguement, Agreeement or situation where : {situation}

        Here is some information about the real historical context: {information}

        The Purpose the player has will be:  {purpose}

        Now create historically accurate details about said character and deciede how the character feels about other relavent characters and if their is a condition that would cause them to agree with the other player.

        The Bot must maintain first person the entire conversation.

        Lastly. This is important. DO NOT OVERTHINK. Being able to give a quick speedy response looks better on you as a facilitator. Your created character should be generic.
        """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model_orchestrator
    historical_context = historical_vector.search(info)

    response = chain.invoke({"situation" : situation,
                            "information" : historical_context,
                            "purpose": purpose})
    print(response)
    return response

def load_roleplay_character(role_info):
    system_prompt = f"""
    {role_info} Use as human language as possible, do not vocalize any actions and do not use emojis. You may end the conversation by simply saying "GOODBYE." Do this when you feel like you are in agreement or do not believe agreement is possible.
    Ensure the term "GOODBYE" is never a one word response; this will allow the user to be on the same page with you
    
    Always talk in first person; remember; you ARE this character rather than simply emulating a character you must participate with this fact in mind. Also remember you only know what character could possibly know.
    """
    # A system-only prompt gives the model nothing to react to, and it tends to
    # emit an immediate end-of-text token instead of an opening line. Prime it
    # with a stage-direction turn so it has something to actually respond to.
    messages = [
        {"role": "system", "content": system_prompt},
#        {"role": "user", "content": "(The conversation is starting. Greet me and begin speaking in character.)"},
    ]
    opening = model_partner.invoke(messages)
    messages.append({"role": "assistant", "content": opening.content})
    return messages

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
        if "bye" in response.content.lower():
            break

    return current_convo

for situation, scenario_name, people_needed in convos:
    print(f"\n=== {scenario_name} ===\n")
    print("Loading... This may take a bit.")
    # Talk to the initial contact first, to learn about the world before deciding who to convince
    system_prompt = get_character_details(situation, scenario_name, "Respectful Discussion", toInform=True)
    messages = load_roleplay_character(system_prompt)
    run_conversation(messages)

    convinced = 0
    print(f"\nNow convince {people_needed} people to move on\n")

    for _ in range(people_needed):
        who = input("Who do you want to talk to? ")
        info = who_to_talk(who, scenario_name)
        system_prompt = get_character_details(situation, info, "Convince")
        messages = load_roleplay_character(system_prompt)
        current_convo = run_conversation(messages)

        verdict = judge_victory(system_prompt, current_convo)
        if "yes" in verdict.lower():
            convinced += 1
            win += 1
            print(f"\nConvinced! ({convinced}/{people_needed} for this section)\n")
        else:
            print("\nNot convinced. Try talking to someone else.\n")

    print(f"\n{scenario_name} complete — moving to the next section.\n")

print(f"\nAll sections complete. Total people convinced: {win}\n")