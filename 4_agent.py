from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent"

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches current weather data for given city
    """
    url = f'https://api.weatherstack.com/current?access_key={os.getenv("WEATHERSTACK_API")}&query={city}'


    response = requests.get(url)

    return response.json()


llm = ChatGroq(model="llama3-70b-8192")

prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

agent = create_react_agent(
    llm=llm,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)


agent_executor = AgentExecutor(
    agent =agent,
    tools=[search_tool,get_weather_data],
    verbose=True,
    max_iterations=10
)


response=agent_executor.invoke({"input":"Identify birth place of kalpana chawla then tell me the weather of that place"})
print(response)

print(response['output'])