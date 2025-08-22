from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = "Sequential LLM chain"
load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']

)


prompt2 = PromptTemplate(
    template="generate a 5 poointer summary from the following text \n {text}",
    input_variables=['text']

)


model1 = ChatGroq(model="Gemma2-9b-It")
model2 = ChatGroq(model="llama3-70b-8192")


parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    "tags":['llm app',"report generation","summarization"],
    "metadata": {"model1": "Gemma2-9b-It","parser":"StrOutputParser"}
}

result = chain.invoke({"topic":"Unemployment in India"},config=config)

print(result)