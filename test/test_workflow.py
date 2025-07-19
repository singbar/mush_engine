"""
Test function to invoke researcher agent and script writer workflow in sequence.
Assumes:
- recommend_viral_dog_reels() -> List[Dict[str, Any]] with keys including idea_id, hook, etc.
- script_writer_workflow.generate_scripts_parallel() returns Dict[str, ScriptResult]
- Optional: rank_scripts() available for scoring.
"""

import json
from langchain_local.researcher_agent import recommend_viral_dog_reels
from langchain_local.script_writer_workflow import (
    generate_scripts_parallel,
    rank_scripts,
    scripts_to_text  # if you included this helper
)

def test_full_workflow():
    query = "viral dog reels"
    dog_profile = (
        "Cute and loyal German Shepherd named Mushroom (Musher, Mushy), 4 y/o, Seattle. "
        "Enjoys homemade treats, indoor ball play, new places, howling at firetrucks, "
        "chasing rabbits & squirrels, and barking at visitors."
    )

    # Step 1: Get structured reel ideas
    ideas = recommend_viral_dog_reels(query, dog_profile)

    # Filter and sanity check
    filtered_ideas = [
        i for i in ideas
        if isinstance(i, dict) and "idea_id" in i and "hook" in i
    ]
    if not filtered_ideas:
        raise RuntimeError("No valid ideas returned from researcher agent.")

    print(f"Fetched {len(filtered_ideas)} ideas.")
    print("Idea IDs:", [i['idea_id'] for i in filtered_ideas])

    # Step 2: Generate scripts directly from structured ideas
    script_results = generate_scripts_parallel(filtered_ideas, dog_profile)

    # Step 3 (Optional): Rank scripts
    leaderboard = rank_scripts(script_results)

    # Print leaderboard summary
    print("\n=== Leaderboard ===")
    for row in leaderboard:
        print(f"{row['rank']}. {row['idea_ref']} score={row['score']} hookQ={row.get('c_hook_quality_score')}")

    # Step 4: Detailed output per script
    print("\n=== Scripts ===")
    for row in leaderboard:
        ref = row["idea_ref"]
        res = script_results[ref]
        print("=" * 80)
        print(f"RANK {row['rank']} | REF {ref} | SCORE {row['score']}")
        if res.warnings:
            print("Warnings:", "; ".join(res.warnings))
        if res.script_json:
            print(json.dumps(res.script_json, ensure_ascii=False, indent=2))
        else:
            print("ERROR:", res.error)
            if res.raw_text:
                print("RAW OUTPUT (truncated):", res.raw_text[:400])

    return script_results, leaderboard

if __name__ == "__main__":
    test_full_workflow()
