### LangChain imports ###
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.agents import create_csv_agent, Tool, AgentExecutor, ZeroShotAgent
from langchain import llms, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
### LangChain imports ###

### Speech imports ###
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound  
### Speech imports ###

### Basic imports ###
import json
import os
import re
from datetime import datetime
### Basic imports ###

### Speech recognizer init and file read ###
r = sr.Recognizer()
f = open('KnowledgeDocument(pan_card_services).txt', "r",  encoding="utf8")
txt = f.read()
### Speech recognizer init and file read ###

### Loading credentials from local and setting in env variables ###
creds = json.load(open('creds.json'))
os.environ["OPENAI_API_KEY"] = creds["openai_key"] # Please update the OpenAI key in creds.json
os.environ["GOOGLE_API_KEY"] = creds["googleapi_key"]
os.environ["GOOGLE_CSE_ID"] = creds["googlecse_id"]
### Loading credentials from local and setting in env variables ###

### Cleaning data with Regex and removing questions ###
txt = re.sub(r"[^.?]+\?","",txt)
txt = re.sub(r"\*{2}","",txt)
### Cleaning data with Regex and removing questions ###

### Text tokenization and embedding ###
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n\n")
texts = text_splitter.split_text(txt)
embeddings = OpenAIEmbeddings()
### Text tokenization and embedding ###

### DB persistence of embeddings ###
persist_directory = 'db'
docsearch = Chroma.from_texts(
    texts,
    embeddings,
    persist_directory = persist_directory,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]
    )
retriever=docsearch.as_retriever()
### DB persistence of embeddings ###

### Preparing the Google search API tool ###
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
### Preparing the Google search API tool ###

### Creating memory object for both agents ###
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
### Creating memory object for both agents ###

### Preparing prompt for qa with sources chain ###
prefix = """Have a conversation with a human, answering the following questions from the source data. Do not create something if you do not know. In that case, tell "Don't know the answer"""
suffix = """Begin!
{summaries}
Question: {question}
"""

prompt = PromptTemplate(
    template=prefix+suffix ,
    input_variables=["summaries", "question"]
    )
### Preparing prompt for qa with sources chain ###

### Creating the qa with sources chain ###
qa_chain = load_qa_with_sources_chain(llm = OpenAI(temperature=0), chain_type="stuff",
                                      prompt=prompt
                                      )
chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=retriever, return_source_documents = False)
### Creating the qa with sources chain ###

### Preparing prompt for zero shot agent ###
prefix = """Have a conversation with a human, answering the following questions as best as you can. You have access to following tools:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
    )
### Preparing prompt for zero shot agent ###

### Creating the zero shot agent chain ###
llm_chain = LLMChain(llm=llms.OpenAI(temperature=0), prompt=prompt)
agent_zs = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent_zs, tools=tools, verbose=True, memory=memory, max_iterations=5) # agent breaks after 5 iterations
### Creating the zero shot agent chain ###

# Reusing prompt variable for infinite loop #
prompt = ""

### speak function takes text and speaks over speaker ###
def speak(text):

    myobj = gTTS(text=text, lang='en', slow=False)
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    myobj.save("temp_{}.mp3".format(date_string))
    playsound("temp_{}.mp3".format(date_string))
    os.remove("temp_{}.mp3".format(date_string))
### Speak function takes text and speaks over speaker ###

### Infinite loop for a conversation session, breaks with exit or bye ###
while True:

    test_speech = ""

    ### Try except block to handle user inputs over microphone ###
    with sr.Microphone() as source:
    
        try:
            
            print("ask your question - ")
            speak("ask your question")

            r.adjust_for_ambient_noise(source, duration=0.2)                 
            # Listens for the user's input
            audio_data = r.listen(source)            
            test_speech = r.recognize_google(audio_data)
            
            print(test_speech)

        except sr.exceptions.UnknownValueError:
            
            print("Didn't receive any input!")
            continue

        except Exception as e:

            print("Error occured!")
            break
    ### Try except block to handle user inputs over microphone ###

    prompt = test_speech.lower() # input('Type your question - ').lower()

    # Break loop if encountered bye or exit #
    if('exit'  in prompt or 'bye' in prompt):
        
        print("Goodbye! Session closed!")
        break
    # Break loop if encountered bye or exit #

    # Sending question to qa with sources chain
    result = chain({"question": prompt})
    
    ### If textual data fails, zero shot agent triggers google search ###
    if("Don't know" not in result["answer"] and 'not available' not in result["answer"] and 'not specified' not in result["answer"]):
        
        print(result["answer"])
        speak(result["answer"])
        # adding question and answer to shared memory #
        memory.chat_memory.add_user_message(prompt)
        memory.chat_memory.add_ai_message(result["answer"])
        # adding question and answer to shared memory #
    
    else:

        # Sending question to qa with sources chain
        result = agent_chain.run(prompt)
        print(result)
        speak(result)
    ### If textual data fails, zero shot agent triggers google search ###

### Infinite loop for a conversation session, breaks with exit or bye ###