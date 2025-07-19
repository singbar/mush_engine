from temporalio import activity
from langchain_local.researcher_agent import recommend_viral_dog_reels
from langchain_local.agents import initialize_agent, AgentType
from langchain_local.llms import OpenAI
from langchain_local.prompts import PromptTemplate
from langchain_local.tools import Tool
from googleapiclient.discovery import build
from langchain_local.memory import ConversationBufferMemory

@activity.defn
def run_research_agent(query: str, dog_profile: str) -> str:
    query = "viral dog reels"
    dog_profile = "Cute and loyal german shepherd, 4 years old, named mushroom. Lives in seattle washington. Goes by musher or mushy. Has a cantonese mom and american dad. Enjoys homemade treats, playing with a ball indoors, going to new places, howling at firetrucks, chasing rabbits/squirrels, and barking at anyone who comes near the house."
    results = recommend_viral_dog_reels(query, dog_profile)
    print("Recommended viral dog reels:")
    print(results)
    return results