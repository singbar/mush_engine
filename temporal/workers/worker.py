"""
Local Temporal worker definition for viral dog reel project.
"""
import asyncio
from temporalio.worker import Worker
from temporal.activities import conduct_reseearch
from temporal.activities import generate_script
from temporal.workflows import generate_scripts


async def main():
    # Create and run the worker

    async with Worker(
        # Connect to local Temporal server
        "localhost:7233",
        task_queue="dog_reel_task_queue",
        workflows=[generate_scripts.generate_dog_reel_scripts_workflow],
        activities=[conduct_reseearch.run_research_agent, generate_script.generate_scripts_activity],
    ):
        print("Worker started. Listening for tasks...")
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
