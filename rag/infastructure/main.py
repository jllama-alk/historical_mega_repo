from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="qwen3.5:9b")

template = """
You are a roleplay facilitator, you are in charge of giving a student the needed information about his or her character. This information will be used as a system prompt for another large language model, so it must be clear and conscise. 

Here is the situation. Historical Arguement or Agreeement where : {situation}

Here is some information about the character {information}

Now create historically accurate details about said character and deciede how the character feels about other relavent characters and if their is a condition that would cause them to agree with the other player
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

response = chain.invoke({"situation" : "Hey so this is actually just a test. just say  >I heard you loud and clear. Don't overthink it", "information" : []})

print(response)