"""
LangChain agent setup for recommending 3-5 viral dog reels (long-context capable).
"""

import os
import json
import time
import math
import datetime as dt
from typing import List, Dict, Any, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# IMPORTANT: Modern import path (adjust if your LangChain version differs)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # fallback for older installations
    from langchain.chat_models import ChatOpenAI  # type: ignore

from pydantic import BaseModel, Field, ValidationError

# ==============================
# Configuration / Environment
# ==============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

MODEL_NAME = os.getenv("REELS_MODEL_NAME", "gpt-4o")         # replace if you have a longer context variant like gpt-4o-128k
MAX_TREND_ITEMS = 25
IDEA_COUNT_MIN = 3
IDEA_COUNT_MAX = 5

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")
if YOUTUBE_API_KEY == "" or not YOUTUBE_API_KEY:
    raise EnvironmentError("Set YOUTUBE_API_KEY (export YOUTUBE_API_KEY=...).")

# ==============================
# Models / Schemas
# ==============================

class Idea(BaseModel):
    idea_id: str
    hook: str
    core_concept: str
    trend_leverage: str
    shot_list: List[str] = Field(min_items=1, max_items=10)
    on_screen_text: List[str]
    audio_suggestion: str
    caption: str
    hashtags: List[str] = Field(min_items=3, max_items=15)
    virality_rationale: str
    effort_level: str
    _index: Optional[int] = None
    _generated_at: Optional[str] = None
    _warnings: Optional[str] = None

# ==============================
# Trend Fetching
# ==============================

def fetch_trending_dog_shorts(
    api_key: str,
    query: str = "dog #shorts",
    order: str = "date",
    max_results: int = 25
) -> List[Dict[str, Any]]:
    youtube = build("youtube", "v3", developerKey=api_key)
    req = youtube.search().list(
        part="snippet",
        maxResults=min(max_results, 50),
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
    lines = []
    for i, t in enumerate(trends):
        published = (t.get("publishedAt") or "")[:19].replace("T", " ")
        title = (t.get("title") or "")[:120].replace("\n", " ")
        lines.append(f"{i+1}. [{published}] {title} (vid:{t.get('videoId')})")
    return "\n".join(lines)

# ==============================
# Tool Wrapping
# ==============================

def search_viral_dog_reels(query: str) -> str:
    try:
        results = fetch_trending_dog_shorts(YOUTUBE_API_KEY, query=query, max_results=MAX_TREND_ITEMS)
    except HttpError as e:
        return f"ERROR fetching YouTube data: {e}"
    if not results:
        return "No results."
    block = summarize_trends(results)
    return f"Recent Dog Shorts (YouTube) for query='{query}':\n{block}"

search_tool = Tool(
    name="SearchViralDogReels",
    func=search_viral_dog_reels,
    description="Pull a recent list of dog-related short videos (trend signals). Input: short query string."
)

# ==============================
# Prompt Template
# ==============================

BASE_IDEA_PROMPT = """You are a senior short-form social media growth strategist for pet accounts. You conduct thorough research to identify current trends and leverage them for maximum engagement.
Craft {idea_min}-{idea_max} HIGH-POTENTIAL, trend-aware Reel/TikTok concepts for the dog profile, with an emphasis on following trends emerging in the past one week.

Dog Profile:
{dog_profile}

Recent Trend Signals (compressed list):
{trend_block}

OUTPUT RULES:
Return ONLY a top-level JSON array of idea objects.
Each object keys:
- idea_id (kebab-case slug)
- hook (<12 words, no surrounding quotes)
- core_concept (1–2 sentences)
- trend_leverage (reference indices or titles from trend list)
- shot_list (3–6 imperative concise shots)
- on_screen_text (<= len(shot_list); each <= 40 chars)
- audio_suggestion (style or meme audio name)
- caption (engaging line; no leading/trailing quotes)
- hashtags (6–10 items; mix general, breed/local, trend-specific)
- virality_rationale (<=200 chars, why it can perform)
- effort_level ("low" | "medium" | "high")

Strategy:
- Vary angles: comedic, POV, emotional, challenge, transformation, anticipation, cultural fusion (Cantonese mom + American dad), locale (Seattle), behaviors (howling at firetrucks, chasing squirrels, ball obsession).
- Ensure at least one *very low effort* single continuous style idea.
- Avoid duplicating identical trend references.
- Use diversity in hook structure (question, POV, suspense, surprise).
- Focus on *current* meme/format signals implied by recency.

PRIORITY ORDER (most important first):
1. Exploit *trend mechanics* (format, audio style, meme structure, pacing cues).
2. Introduce a *novel twist* using the dog ONLY IF it strengthens the trend’s engagement loop.
3. Minimize generic pet descriptors; avoid repeating the same trait across ideas unless the trend specifically requires it.
DO NOT: produce ideas that could stand without referencing the provided trend list.
Every idea must cite the specific trend indices powering it.

If an idea does not clearly leverage a listed trend mechanic, you must discard and replace it before output.


Return ONLY JSON. No commentary.
"""
#==============================
# Memory & LLM
# ==============================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.7,
    max_tokens=1800,
    request_timeout=120
)

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# ==============================
# Helpers
# ==============================

def estimate_tokens(text: str) -> int:
    # Rough heuristic (4 chars/token)
    return math.ceil(len(text) / 4)

def backoff_retry(callable_fn, retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            return callable_fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))

# ==============================
# Core Generation
# ==============================

def recommend_viral_dog_reels(
    query: str,
    dog_profile: str,
    idea_min: int = IDEA_COUNT_MIN,
    idea_max: int = IDEA_COUNT_MAX,
    use_agent_for_retrieval: bool = False
) -> List[Dict[str, Any]]:
    """
    Steps:
      1. Retrieve trend context (tool direct or via agent).
      2. Build structured prompt.
      3. Invoke LLM.
      4. Parse & validate JSON ideas.
    """
    if use_agent_for_retrieval:
        # Agent path (make sure we pass a string)
        retrieval_prompt = f"Fetch recent dog short-form video trends for query: {query}"
        trend_context_text = agent.run(retrieval_prompt)
    else:
        trend_context_text = search_viral_dog_reels(query)

    if trend_context_text.startswith("ERROR"):
        raise RuntimeError(trend_context_text)

    # Truncate if extremely long
    if estimate_tokens(trend_context_text) > 6000:
        # crude trim; a more robust approach would summarise
        trend_context_text = "\n".join(trend_context_text.splitlines()[:300])

    prompt_str = BASE_IDEA_PROMPT.format(
        idea_min=idea_min,
        idea_max=idea_max,
        dog_profile=dog_profile.strip(),
        trend_block=trend_context_text[:12000]
    )

    if isinstance(prompt_str, list):  # Defensive (should never happen)
        prompt_str = "\n".join(map(str, prompt_str))

    def _invoke():
        # Use .invoke for modern API
        return llm.invoke(prompt_str).content

    completion = backoff_retry(_invoke)

    # Parsing JSON
    ideas_raw: List[Dict[str, Any]]
    try:
        ideas_raw = json.loads(completion)
        if not isinstance(ideas_raw, list):
            raise ValueError("Top-level JSON is not a list.")
    except Exception:
        # Attempt substring extraction
        try:
            start = completion.index('[')
            end = completion.rindex(']') + 1
            ideas_raw = json.loads(completion[start:end])
            if not isinstance(ideas_raw, list):
                raise ValueError
        except Exception as e:
            raise ValueError(f"Failed to parse JSON ideas: {e}\nRaw Output:\n{completion}")

    clean: List[Dict[str, Any]] = []
    for i, idea_obj in enumerate(ideas_raw):
        if not isinstance(idea_obj, dict):
            continue
        idea_obj["_index"] = i
        idea_obj["_generated_at"] = dt.datetime.utcnow().isoformat()

        # Validate with Pydantic, capturing missing keys as warnings
        try:
            _ = Idea(**{k: v for k, v in idea_obj.items() if not k.startswith("_")})
        except ValidationError as ve:
            idea_obj["_warnings"] = f"Validation issues: {ve.errors()}"
        clean.append(idea_obj)

    return clean

# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    query = "viral dog reels"
    dog_profile = (
        "Name: Mushroom (Musher, Mushy). Breed: German Shepherd. Age: 4. "
        "Locale: Seattle. Traits: loyal, playful, ball-obsessed, howls at firetrucks, "
        "chases rabbits & squirrels, protective door bark, enjoys homemade treats. "
        "Cultural fusion: Cantonese mom + American dad."
    )
    ideas = recommend_viral_dog_reels(query, dog_profile)
    print(json.dumps(ideas, indent=2))