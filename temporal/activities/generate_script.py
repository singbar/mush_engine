"""
Temporal activity to generate scripts for viral dog reels using script_writer_workflow.
"""
from temporalio import activity
from langchain_local.script_writer_workflow import generate_scripts_parallel

@activity.defn
def generate_scripts_activity(reel_ideas: list, dog_profile: str) -> dict:
    reel_ideas = ["Mushroom's Seattle Adventures: From Pike Place Market to the Gum Wall!", "Mushroom's Favorite Treats: Homemade Recipes and Taste Tests! #dogtreats #homemade #yum #dogapproved","Mushroom's Howling Sessions: Firetruck Edition! #howling #firetruck #doglife #funny", "Mushroom's Travel Vlog: Exploring New Places with My Humans! #traveldog #newadventures #explore #doglife""Mushroom's Squirrel Chasing Chronicles! #squirrelchaser #doglife #funny #cute #squirrelpower"
    ]
    dog_profile = "Cute and loyal german shepherd, 4 years old, named Mushroom. Lives in Seattle, Washington. Goes by Musher or Mushy. Enjoys homemade treats, playing with a ball indoors, going to new places, howling at firetrucks, chasing rabbits/squirrels, and barking at anyone who comes near the house."
    
    scripts = generate_scripts_parallel(reel_ideas, dog_profile)
    for idea, script in scripts.items():
        print(f"Idea: {idea}\nScript: {script}\n")
    return generate_scripts_parallel(reel_ideas, dog_profile)
