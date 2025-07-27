import json
from langchain_local.researcher_agent import recommend_viral_dog_reels
from langchain_local.script_writer_workflow import (
    generate_scripts_parallel,
    rank_scripts,
    scripts_to_text
)


def test_full_workflow():
    """
    Run the end-to-end pipeline for generating, ranking, and displaying viral dog reel scripts.

    Workflow:
        1. Fetch trending reel ideas for the given dog profile.
        2. Filter and validate the returned ideas.
        3. Generate full scripts (in parallel) based on these ideas.
        4. Rank the generated scripts using custom scoring.
        5. Print a leaderboard and the detailed script outputs.
    
    Returns:
        tuple:
            - script_results (dict): Mapping of idea reference -> script generation results.
            - leaderboard (list[dict]): Ranked list of ideas with associated scores.
    """
    query = "viral dog reels"
    dog_profile = (
        "Cute and loyal German Shepherd named Mushroom (Musher, Mushy), 4 y/o, Seattle. "
        "Enjoys homemade treats, indoor ball play, new places, howling at firetrucks, "
        "chasing rabbits & squirrels, and barking at visitors."
    )

    # Step 1: Fetch structured viral reel ideas via the researcher agent
    try:
        ideas = recommend_viral_dog_reels(query, dog_profile)
    except Exception as e:
        # Fail gracefully: capture the exception and return empty results
        print(f"[WARN] Researcher agent failed to return ideas: {e}")
        ideas = []

    # Step 2: Validate and filter results (don't crash on malformed data)
    filtered_ideas = [
        i for i in ideas
        if isinstance(i, dict) and "idea_id" in i and "hook" in i
    ]
    if not filtered_ideas:
        print("[WARN] No valid ideas returned. Skipping script generation.")
        return {}, []

    print(f"Fetched {len(filtered_ideas)} valid ideas.")
    print("Idea IDs:", [i['idea_id'] for i in filtered_ideas])

    # Step 3: Generate scripts in parallel for the valid ideas
    script_results = generate_scripts_parallel(filtered_ideas, dog_profile)

    # Step 4: Rank the generated scripts
    leaderboard = rank_scripts(script_results)

    # Step 5: Print leaderboard and detailed script outputs
    print("\n=== Leaderboard ===")
    for row in leaderboard:
        print(f"{row['rank']}. {row['idea_ref']} score={row['score']} hookQ={row.get('c_hook_quality_score')}")

    print("\n=== Scripts ===")
    for row in leaderboard:
        ref = row["idea_ref"]
        res = script_results[ref]
        print("=" * 80)
        print(f"RANK {row['rank']} | REF {ref} | SCORE {row['score']}")
        if res.warnings:
            print("Warnings:", "; ".join(res.warnings))
        if res.script_json:
            # Pretty-print the generated script as JSON
            print(json.dumps(res.script_json, ensure_ascii=False, indent=2))
        else:
            # Show fallback debug info if the script failed
            print("ERROR:", res.error)
            if res.raw_text:
                print("RAW OUTPUT (truncated):", res.raw_text[:400])

    return script_results, leaderboard


if __name__ == "__main__":
    test_full_workflow()
