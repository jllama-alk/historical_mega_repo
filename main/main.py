from time import sleep
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from vectordb.vector import HistoricalVector
import threading

stop = False
def timer():
    while not stop:
        print("x")
        sleep(5)

historical_vector = HistoricalVector()

model = OllamaLLM(model="gemma4:e4b", keep_alive=0)

template = """
You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise. 

Here is the situation. Historical Arguement, Agreeement or situation where : {situation}

Here is some information about the real historical context: {information}

Now create historically accurate details about said character and deciede how the character feels about other relavent characters and if their is a condition that would cause them to agree with the other player.

Lastly. This is important. DO NOT OVERTHINK. Being able to give a quick speedy response looks better on you as a facilitator. Your created character should be generic.
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
thread1 = threading.Thread(target=timer)
thread1.start()
historical_context = historical_vector.search("A group conversation between a native Algerian, a Morrocan, and a french settler")
print(historical_context, "\n")
print("asking qwen\n")

response = chain.invoke({"situation" : "An algerian, a morrocan and a french settler happen to be sitting in the same cafe in Casablanca. You are the Moroccan",
                          "information" : historical_context})

print(historical_context, "\n")
print(response)
stop = True
thread1.join()