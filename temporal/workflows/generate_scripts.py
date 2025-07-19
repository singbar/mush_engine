"""
Temporal workflow to connect research and script writing activities for viral dog reels.
"""
from temporalio import workflow
from temporal.activities.conduct_reseearch import run_research_agent
from langchain_local.Script_writer_agent import generate_scripts_parallel

@workflow.defn
async def generate_dog_reel_scripts_workflow(query: str, dog_profile: str) -> dict:
    """
    Workflow to generate viral dog reel ideas and turn them into scripts.
    Args:
        query (str): The search query for viral dog reels.
        dog_profile (str): Description of the dog for personalization.
    Returns:
        dict: Mapping of reel idea to generated script.
    """
    # Step 1: Get viral dog reel ideas from research agent
    ideas_text = await workflow.execute_activity(
        run_research_agent,
        query,
        dog_profile,
        schedule_to_close_timeout=300
    )

    # Parse ideas from the agent's output (assuming numbered list)
    reel_ideas = []
    for line in ideas_text.splitlines():
        if line.strip() and line[0].isdigit() and '.' in line:
            idea = line.split('.', 1)[1].strip()
            reel_ideas.append(idea)

    # Step 2: Generate scripts for each idea using the activity
    from temporal.activities.generate_script import generate_scripts_activity
    scripts = await workflow.execute_activity(
        generate_scripts_activity,
        reel_ideas,
        dog_profile,
        schedule_to_close_timeout=600
    )
    return scripts
