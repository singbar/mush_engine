"""
LangChain agent setup for recommending 3–5 viral dog reels (long-context capable).

This script automates the generation of trend-aware social media content ideas
(Reels/TikTok) for a specific dog profile by:
  - Fetching current trending dog videos from YouTube (#shorts).
  - Summarizing those trends for context.
  - Prompting an LLM (ChatGPT via LangChain) to create structured, validated JSON ideas.
  - Validating the ideas with a Pydantic schema for consistency.

Designed for use as either a CLI script or importable module.
"""

import os
import re
import json
import time
import math
import dotenv
import datetime as dt
from typing import List, Dict, Any, Optional

# Google API for YouTube trend scraping
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# LangChain: core tools, agent orchestration, and memory for multi-turn contexts
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain.memory import ConversationBufferMemory

# Import OpenAI chat model (modern path preferred)
from langchain_community.chat_models import ChatOpenAI

# Pydantic for schema validation (ensures output integrity)
from pydantic import BaseModel, Field, ValidationError

# ==============================
# Environment Configuration
# ==============================

#Load environment from dotenv file
dotenv.load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# LLM model configuration (supports long context)
MODEL_NAME = os.getenv("REELS_MODEL_NAME", "gpt-4o")  # Use gpt-4o-128k for ultra-long context if needed

# Default constants controlling behavior
MAX_TREND_ITEMS = 25        # Limit the number of YouTube videos fetched
IDEA_COUNT_MIN = 3           # Minimum ideas generated
IDEA_COUNT_MAX = 5           # Maximum ideas generated

# Validate environment configuration early to avoid runtime surprises
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")
if not YOUTUBE_API_KEY:
    raise EnvironmentError("Set YOUTUBE_API_KEY (export YOUTUBE_API_KEY=...).")

# ==============================
# Data Model for Generated Ideas
# ==============================

class Idea(BaseModel):
    """
    Defines the schema for each viral Reel/TikTok idea returned by the LLM.
    Pydantic ensures that the ideas adhere to expected structure, catching issues like
    missing keys or wrong types.
    """
    idea_id: str                          # Kebab-case slug identifier
    hook: str                             # Short attention-grabbing opener (<12 words)
    core_concept: str                     # 1–2 sentence description of the idea
    trend_leverage: str                   # References to trend indices or titles from trend list
    shot_list: List[str] = Field(min_items=1, max_items=10)  # Step-by-step camera shots
    on_screen_text: List[str]             # Text overlays (aligned with shots)
    audio_suggestion: str                 # Meme, music, or style cue
    caption: str                          # Instagram caption text
    hashtags: List[str] = Field(min_items=3, max_items=15)  # Hashtags (blend of general + trend-specific)
    virality_rationale: str               # Short reasoning on why it will perform
    effort_level: str                     # "low" | "medium" | "high" production effort
    # Metadata fields (not part of schema validation)
    _index: Optional[int] = None
    _generated_at: Optional[str] = None
    _warnings: Optional[str] = None

# ==============================
# Fetch Trending Dog Shorts (YouTube API)
# ==============================

def fetch_trending_dog_shorts(
    api_key: str,
    query: str = "dog #shorts",
    order: str = "date",
    max_results: int = 25
) -> List[Dict[str, Any]]:
    """
    Queries YouTube for trending short-form dog-related videos to use as trend signals.

    Returns:
        List of dictionaries containing video metadata (title, channel, publish date, etc.).
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    req = youtube.search().list(
        part="snippet",
        maxResults=min(max_results, 50),  # API limit: max 50
        q=query,
        type="video",
        order=order
    )
    res = req.execute()

    items = []
    for item in res.get("items", []):
        snippet = item.get("snippet", {})
        items.append({
            "platform": "youtube",
            "title": snippet.get("title"),
            "channelTitle": snippet.get("channelTitle"),
            "videoId": item["id"]["videoId"],
            "publishedAt": snippet.get("publishedAt"),
            "description": snippet.get("description"),
            "query_used": query
        })
    return items

def summarize_trends(trends: List[Dict[str, Any]]) -> str:
    """
    Summarizes YouTube trend items into a human-readable list, which becomes part of the LLM prompt.
    Each line includes:
      - Index number
      - Publish timestamp
      - Video title (truncated for length)
      - Video ID
    """
    lines = []
    for i, t in enumerate(trends):
        published = (t.get("publishedAt") or "")[:19].replace("T", " ")
        title = (t.get("title") or "")[:120].replace("\n", " ")
        lines.append(f"{i+1}. [{published}] {title} (vid:{t.get('videoId')})")
    return "\n".join(lines)

# ==============================
# LangChain Tool: Wrapping Trend Search
# ==============================

def search_viral_dog_reels(query: str) -> str:
    """
    Exposes YouTube trend fetching as a LangChain Tool, so the agent can dynamically call it.
    Returns a formatted text block summarizing recent dog shorts.
    """
    try:
        results = fetch_trending_dog_shorts(YOUTUBE_API_KEY, query=query, max_results=MAX_TREND_ITEMS)
    except HttpError as e:
        return f"ERROR fetching YouTube data: {e}"
    if not results:
        return "No results."
    return f"Recent Dog Shorts (YouTube) for query='{query}':\n{summarize_trends(results)}"

search_tool = Tool(
    name="SearchViralDogReels",
    func=search_viral_dog_reels,
    description="Pull a recent list of dog-related short videos (trend signals). Input: short query string."
)

# ==============================
# Prompt Template for LLM
# ==============================

BASE_IDEA_PROMPT = """... (omitted here for brevity, see original) ..."""

# The prompt:
# - Forces structured JSON output.
# - Guides the model to vary creative angles (POV, comedic, transformation).
# - Enforces citation of trend indices to ensure *trend-driven* content.

# ==============================
# Memory & LLM Setup
# ==============================

# ConversationBufferMemory allows the agent to remember context across multiple turns
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Configured ChatGPT model for generation
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.7,          # Creative but controlled
    max_tokens=1800,           # Generous output space for 3–5 ideas
    request_timeout=120        # Avoid timeouts for longer trend blocks
)

# Agent orchestration: wraps the LLM with the search tool, enabling dynamic retrieval
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # Chain-of-thought capable agent
    verbose=True,                                      # Debug logs to stdout
    memory=memory,
    handle_parsing_errors=True                         # Resilient to malformed responses
)

# ==============================
# Utility Functions
# ==============================

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough heuristic: ~4 characters/token)."""
    return math.ceil(len(text) / 4)

def backoff_retry(callable_fn, retries=3, base_delay=2):
    """
    Executes a function with exponential backoff retries.
    Useful for API calls (OpenAI, YouTube) to handle transient errors.
    """
    for attempt in range(retries):
        try:
            return callable_fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))  # Backoff growth: 2, 4, 8...

import json, re
from langchain.schema import HumanMessage, SystemMessage

def force_json_retry(raw_output: str, dog_profile: str, idea_min: int, idea_max: int) -> list:
    """
    Reprompts the LLM to repair or reformat the ideas as valid JSON.
    Automatically cleans code fences and normalizes hashtags to lists.
    """
    reprompt = (
        f"The last output was invalid. Please return ONLY a JSON array with "
        f"{idea_min}-{idea_max} objects, each with keys: "
        f"idea_id, hook, core_concept, trend_leverage, shot_list, on_screen_text, "
        f"audio_suggestion, caption, hashtags (6–10 as array), virality_rationale, effort_level. "
        f"No markdown fences, no commentary. Output must be strictly valid JSON."
    )
    messages = [
        SystemMessage(content="You are a data generator for viral dog reel ideas."),
        HumanMessage(content=reprompt),
        HumanMessage(content=f"Original raw output:\n{raw_output}")
    ]

    try:
        resp = llm.invoke(messages)
        cleaned = re.sub(r"```(?:json)?", "", resp.content, flags=re.IGNORECASE).strip()
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise RuntimeError(f"Reprompt failed. Output:\n{resp.content}")

        candidate = match.group(0)
        ideas = json.loads(candidate)

        # Normalize hashtags: make sure it's always a list
        for idea in ideas:
            if isinstance(idea.get("hashtags"), str):
                tags = re.split(r"[ ,]+", idea["hashtags"].strip())
                idea["hashtags"] = [t for t in tags if t]

        return ideas
    except Exception as e:
        raise RuntimeError(f"Reprompt failed. Output:\n{resp.content}") from e

def extract_json_array(text: str) -> list:
        """
        Attempts to extract and parse a JSON array from a model's output.
        Handles markdown fences, extra commentary, and partial results.
        Returns an empty list if nothing valid found.
        """
        if not text:
            return []
        
        # Remove common markdown fences like ```json or ```
        cleaned = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()

        # Find first JSON array in the text
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []

        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
        return []

def clean_and_parse_ideas(text: str) -> list:
    """Extracts JSON array, strips code fences, and fixes hashtags."""
    if not text:
        return []

    # Remove markdown code fences like ```json and ```
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Try to find first JSON array
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        return []
    candidate = match.group(0)

    try:
        ideas = json.loads(candidate)
        # Normalize hashtags: ensure each idea has a list
        for idea in ideas:
            if isinstance(idea.get("hashtags"), str):
                # Split by spaces or commas
                tags = re.split(r"[ ,]+", idea["hashtags"].strip())
                idea["hashtags"] = [t for t in tags if t]
        return ideas
    except Exception:
        return []

# ==============================
# Core: Viral Reel Recommendation
# ==============================

def recommend_viral_dog_reels(
    query: str,
    dog_profile: str,
    idea_min: int = IDEA_COUNT_MIN,
    idea_max: int = IDEA_COUNT_MAX,
    use_agent_for_retrieval: bool = True
) -> List[Dict[str, Any]]:
    """
    Orchestrates the full pipeline:
      1. Fetch recent YouTube trend context (via tool or direct function).
      2. Build a structured prompt.
      3. Call the LLM to generate JSON ideas.
      4. Parse, validate, and enrich the ideas with metadata.

    Returns:
        A list of validated viral content ideas, each as a dictionary.
    """
    # Step 1: Retrieve trend signals (direct fetch or through agent chain)
    trend_context_text = (
        agent.run(f"Fetch recent dog short-form video trends for query: {query}")
        if use_agent_for_retrieval else
        search_viral_dog_reels(query)
    )

    if trend_context_text.startswith("ERROR"):
        raise RuntimeError(trend_context_text)

    # Step 2: Truncate extremely long trend blocks (avoid token overflow)
    if estimate_tokens(trend_context_text) > 25000:
        trend_context_text = "\n".join(trend_context_text.splitlines()[:300])

    # Step 3: Assemble final prompt with dynamic parameters
    prompt_str = BASE_IDEA_PROMPT.format(
        idea_min=idea_min,
        idea_max=idea_max,
        dog_profile=dog_profile.strip(),
        trend_block=trend_context_text[:12000]
    )

    # Defensive fallback: ensure prompt is always a string
    if isinstance(prompt_str, list):
        prompt_str = "\n".join(map(str, prompt_str))

    # Step 4: Invoke LLM with retries
    def _invoke():
        return llm.invoke(prompt_str).content

    completion = backoff_retry(_invoke)

    # Step 5: Parse JSON (with retry if the agent output isn't valid JSON)
    # First, try direct parse
    try:
        ideas_raw: List[Dict[str, Any]] = clean_and_parse_ideas(completion)
        if not isinstance(ideas_raw, list):
            raise ValueError("Top-level JSON is not a list.")
    except Exception:
        # Try regex/code-fence extraction
        ideas_raw = extract_json_array(completion)

    # If still no luck, reprompt for clean JSON
    if not ideas_raw:
        ideas_raw = force_json_retry(completion, dog_profile, idea_min, idea_max)


    # Step 6: Validate and enrich ideas with metadata
    clean: List[Dict[str, Any]] = []
    for i, idea_obj in enumerate(ideas_raw):
        if not isinstance(idea_obj, dict):
            continue
        idea_obj["_index"] = i
        idea_obj["_generated_at"] = dt.datetime.utcnow().isoformat()

        # Pydantic validation to catch missing or malformed fields
        try:
            _ = Idea(**{k: v for k, v in idea_obj.items() if not k.startswith("_")})
        except ValidationError as ve:
            idea_obj["_warnings"] = f"Validation issues: {ve.errors()}"
        clean.append(idea_obj)

    return clean

# ==============================
# CLI Entrypoint (Standalone Execution)
# ==============================

if __name__ == "__main__":
    # Example invocation: generate 3–5 ideas for Mushroom the German Shepherd
    query = "viral dog reels"
    dog_profile = (
        "Name: Mushroom (Musher, Mushy). Breed: German Shepherd. Age: 4. "
        "Locale: Seattle. Traits: loyal, playful, ball-obsessed, howls at firetrucks, "
        "chases rabbits & squirrels, protective door bark, enjoys homemade treats. "
        "Cultural fusion: Cantonese mom + American dad."
    )
    ideas = recommend_viral_dog_reels(query, dog_profile)
    print(json.dumps(ideas, indent=2))