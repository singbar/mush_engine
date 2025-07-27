"""
script_writer_workflow.py

Purpose:
--------
Generate detailed, production-ready short-form dog reel scripts (3–6 shots each)
from raw social media ideas, validate the outputs, optionally repair them, and
rank the final scripts using heuristic scoring (shot quality, hooks, editing cues, etc.).

Key Features:
-------------
- Accepts raw string or structured JSON ideas as input.
- Uses OpenAI Chat model (via LangChain) for long-context script generation.
- Parallelized script generation with retries and exponential backoff.
- Schema validation and repair prompts for malformed or incomplete JSON.
- Heuristic scoring and ranking of final scripts (leaderboard format).
- CLI demonstration for local testing.

Required ENV:
    OPENAI_API_KEY

Optional ENV:
    SCRIPT_MODEL_NAME   (default: gpt-4o)
    SCRIPT_MAX_WORKERS  (default: 6)

Author: (Your Project)
"""

from __future__ import annotations
import os
import time
import math
import json
import uuid
import re
import dotenv
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================================================
# LLM Setup (LangChain with OpenAI backend)
# ==================================================
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback for older LangChain installs
    from langchain.chat_models import ChatOpenAI  # type: ignore

from langchain.schema import HumanMessage, SystemMessage

# ==================================================
# Configuration (model, retries, and safety)
# ==================================================

#load from dotenv file
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("SCRIPT_MODEL_NAME", "gpt-4o")
MAX_WORKERS = int(os.getenv("SCRIPT_MAX_WORKERS", "6"))
TEMPERATURE = 0.65
MAX_MODEL_TOKENS = 900               # Token limit per generation
RETRY_ATTEMPTS = 4                   # Attempts for primary generation
BASE_BACKOFF = 2.0                   # Exponential backoff base (2, 4, 8s...)
REPAIR_ATTEMPTS = 1                  # Secondary repair passes (if validation fails)
QUALITY_DEBUG = False                # Debug scoring breakdowns

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")

# ==================================================
# Prompt Templates (System + User Messages)
# ==================================================
SCRIPT_SYSTEM = (
    "You are a senior short-form video director for high-retention pet Reels/TikToks. "
    "You output ONLY valid JSON meeting the requested schema."
)

# Schema description (ensures consistent structure)
SCHEMA_BLOCK = """
Required JSON object keys:
idea_id (string),
summary (<= 200 chars),
hook: { spoken, on_screen },
shots: [ { order(int), duration_s(float), camera(str), framing(str), action(str),
          on_screen_text?(str<=40), narration_or_audio(str), transition_in?(str|null),
          transition_out?(str|null), notes?(str) } ],
broll_inserts?: [ { cue_shot_order(int), action(str), purpose(str) } ],
audio_direction: { type(str), reference?(str), sync_notes?(str) },
editing_notes (string),
cta (string),
caption (string),
hashtags (array 6-10 strings),
effort_level ("low"|"medium"|"high"),
gear_checklist (array of strings),
publishing_tips (string),
rationale (<=180 chars),
generated_at (ISO8601)
"""

# Full user-facing instruction prompt (merged with dog profile & idea)
SCRIPT_USER_TEMPLATE = """Dog Profile:
{dog_profile}

Original Idea (JSON or text):
{idea_json}

TASK:
Transform the idea into a fully execution-ready script (not just a paraphrase).
Include specific pacing, camera moves, anticipation beats, pattern interrupts,
and editing cues for professional-style Reels/TikToks.

OUTPUT RULES:
Return ONLY one JSON object using this schema. No markdown, no commentary.
{schema_block}

Constraints:
- 3–6 primary shots (condense verbose sections as needed).
- Spoken hook <12 words; can differ from on-screen hook for contrast.
- Each action must be specific (avoid vague "dog does something").
- Include at least one anticipation or suspense payoff moment.
- Include at least one pattern interrupt (zoom, b-roll, jump cut, speed change, etc.).
- Hashtags: 6–10 (mix general, breed, locale, and trending).
- Avoid emojis; no trailing periods on hashtags.
- rationale <=180 chars.
- editing_notes must reference at least 1 specific editing technique (cut, zoom, color, etc.).
- JSON must be strictly valid (no trailing commas, all required keys).
"""

# ==================================================
# Data Structure for Script Results
# ==================================================
@dataclass
class ScriptResult:
    """
    Captures the full outcome of a single script generation attempt:
      - References to input idea and type.
      - Script JSON (if successful).
      - Raw LLM text output (for debugging/repair).
      - Errors, latency, attempts, and whether a repair prompt was used.
      - Final quality score, score components, and any validation warnings.
    """
    idea_ref: str
    idea_source_type: str
    script_json: Dict[str, Any] | None
    raw_text: str | None
    error: str | None
    model: str
    latency_s: float
    attempts: int
    repaired: bool
    score: float | None = None
    score_components: Dict[str, float] | None = None
    warnings: List[str] | None = None

# ==================================================
# LLM Instance (Global)
# ==================================================
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_MODEL_TOKENS
)

# ==================================================
# Utility Functions
# ==================================================
def estimate_tokens(text: str) -> int:
    """Estimate token count (heuristic: ~4 characters per token)."""
    return math.ceil(len(text) / 4)

def truncate(text: str, max_tokens: int, margin: int = 200) -> str:
    """
    Truncate text if token estimate (plus margin) exceeds limit.
    Prevents exceeding context window for long ideas.
    """
    if estimate_tokens(text) + margin <= max_tokens:
        return text
    target_chars = (max_tokens - margin) * 4
    return text[:target_chars] + "... [TRIMMED]"

def iso_now() -> str:
    """Return current UTC time in ISO8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def safe_get(d: Dict[str, Any], path: str, default=None):
    """Safely retrieve nested dict values using dotted paths."""
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ==================================================
# Prompt Builder (Combines System & User Messages)
# ==================================================
def build_script_prompt(idea: Union[str, Dict[str, Any]], dog_profile: str) -> Tuple[List[Any], str]:
    """
    Construct a structured LangChain prompt for the LLM based on:
      - The original idea (string or dict).
      - The dog's profile (traits, locale, quirks).

    Returns:
        - List of LangChain messages (System + Human).
        - Reference string for the idea (for result tracking).
    """
    if isinstance(idea, dict):
        # Only include essential fields for context (avoid clutter)
        compact_keys = {
            "idea_id", "hook", "core_concept", "trend_leverage",
            "shot_list", "on_screen_text", "audio_suggestion",
            "caption", "hashtags"
        }
        compact = {k: idea[k] for k in idea if k in compact_keys}
        if "idea_id" not in compact:
            compact["idea_id"] = idea.get("idea_id") or f"idea-{uuid.uuid4().hex[:6]}"
        idea_json = json.dumps(compact, ensure_ascii=False)
        idea_ref = compact["idea_id"]
        source_type = "dict"
    else:
        # Treat freeform strings as-is
        idea_json = idea.strip()
        idea_ref = f"raw-{uuid.uuid4().hex[:6]}"
        source_type = "string"

    # Ensure token count is safe
    idea_json = truncate(idea_json, 3000)

    # Final user message assembly
    user_message = SCRIPT_USER_TEMPLATE.format(
        dog_profile=dog_profile.strip(),
        idea_json=idea_json,
        schema_block=SCHEMA_BLOCK.strip()
    )
    return [SystemMessage(content=SCRIPT_SYSTEM), HumanMessage(content=user_message)], f"{idea_ref}::{source_type}"

# ==================================================
# JSON Validation and Repair Prompts
# ==================================================
REQUIRED_TOP_KEYS = {
    "idea_id","summary","hook","shots","audio_direction",
    "editing_notes","cta","caption","hashtags","effort_level",
    "gear_checklist","publishing_tips","rationale","generated_at"
}

def validate_script(script: Dict[str, Any]) -> List[str]:
    """
    Validate generated script structure.
    Returns a list of warnings for any structural or semantic issues.
    """
    warnings = []
    # Required keys
    missing = REQUIRED_TOP_KEYS - set(script.keys())
    if missing:
        warnings.append(f"Missing keys: {sorted(missing)}")

    # Hook must have spoken & on_screen
    hook = script.get("hook", {})
    if not isinstance(hook, dict) or any(k not in hook for k in ("spoken","on_screen")):
        warnings.append("hook must include spoken & on_screen")

    # Shots (3–6, each with action & duration)
    shots = script.get("shots", [])
    if not isinstance(shots, list) or not shots:
        warnings.append("shots missing or empty")
    else:
        if not (3 <= len(shots) <= 6):
            warnings.append("shot count not in 3–6 range")
        for s in shots:
            if "action" not in s:
                warnings.append("shot missing action")
            if "duration_s" not in s:
                warnings.append("shot missing duration_s")

    # Hashtags (6–10)
    hashtags = script.get("hashtags", [])
    if not (6 <= len(hashtags) <= 10):
        warnings.append("hashtags count not 6–10")

    # Rationale length
    rationale = script.get("rationale","")
    if len(rationale) > 180:
        warnings.append("rationale >180 chars")

    # Effort level must be valid
    effort = script.get("effort_level","")
    if effort not in {"low","medium","high"}:
        warnings.append("effort_level invalid")

    return warnings

def build_repair_prompt(original_raw: str, warnings: List[str]) -> List[Any]:
    """
    Build a prompt instructing the LLM to repair an invalid JSON output.
    Provides the list of issues to fix and asks for a corrected single JSON object.
    """
    repair_instructions = (
        "The previous JSON did not meet validation. Fix ONLY the issues below, "
        "retain all good content, and return a SINGLE corrected JSON object.\n"
        f"Issues: {warnings}\n"
        "Return ONLY valid JSON. No commentary."
    )
    return [
        SystemMessage(content=SCRIPT_SYSTEM),
        HumanMessage(content=repair_instructions),
        HumanMessage(content=original_raw)
    ]

# ==================================================
# Core Generation for a Single Script
# ==================================================
def generate_single_script(
    idea: Union[str, Dict[str, Any]],
    dog_profile: str,
    repair_attempts: int = REPAIR_ATTEMPTS
) -> ScriptResult:
    """
    Generate a single script for a given idea and dog profile.
    Handles:
      - Initial LLM call (with retries).
      - Basic JSON extraction.
      - Validation warnings.
      - Optional repair prompt if validation fails.
    Returns a ScriptResult object with script JSON, metadata, and warnings.
    """
    messages, idea_ref = build_script_prompt(idea, dog_profile)
    start = time.time()
    attempts = 0
    raw_text, script_json, error = None, None, None
    repaired = False

    # Retry loop for generation
    for attempt in range(RETRY_ATTEMPTS):
        attempts = attempt + 1
        try:
            resp = llm.invoke(messages)
            raw_text = resp.content.strip()

            # Extract JSON portion (if extra text slipped through)
            if not raw_text.startswith("{") and "{" in raw_text and "}" in raw_text:
                raw_text = raw_text[raw_text.index("{"):raw_text.rindex("}")+1]

            script_json = json.loads(raw_text)
            if isinstance(script_json, dict):
                script_json.setdefault("generated_at", iso_now())
                break
            else:
                error = "Top-level JSON is not an object."
        except Exception as e:
            error = f"Gen attempt {attempt+1} error: {e}"
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(BASE_BACKOFF * (2 ** attempt))  # Exponential backoff
            else:
                break

    # Validate the result (collect warnings)
    warnings = validate_script(script_json or {})

    # Optional repair pass if warnings exist
    if warnings and repair_attempts > 0:
        repaired = True
        repair_msgs = build_repair_prompt(raw_text or "{}", warnings)
        try:
            resp = llm.invoke(repair_msgs)
            repaired_raw = resp.content.strip()
            if not repaired_raw.startswith("{") and "{" in repaired_raw and "}" in repaired_raw:
                repaired_raw = repaired_raw[repaired_raw.index("{"):repaired_raw.rindex("}")+1]
            repaired_json = json.loads(repaired_raw)
            r_warnings = validate_script(repaired_json)
            if not r_warnings:
                script_json = repaired_json
                raw_text = repaired_raw
                warnings = []
            else:
                warnings = r_warnings
        except Exception as e:
            warnings.append(f"Repair failed: {e}")

    latency = time.time() - start

    return ScriptResult(
        idea_ref=idea_ref,
        idea_source_type=idea_ref.split("::")[-1],
        script_json=script_json,
        raw_text=raw_text,
        error=error,
        model=MODEL_NAME,
        latency_s=latency,
        attempts=attempts,
        repaired=repaired,
        warnings=warnings
    )

# ==================================================
# Parallel Script Generation (Batch Mode)
# ==================================================
def generate_scripts_parallel(
    reel_ideas: Iterable[Union[str, Dict[str, Any]]],
    dog_profile: str,
    max_workers: int = MAX_WORKERS,
    progress_callback: Optional[Callable[[ScriptResult], None]] = None
) -> Dict[str, ScriptResult]:
    """
    Generate scripts for multiple ideas concurrently using a thread pool.
    Each idea is passed to `generate_single_script`.

    Args:
        reel_ideas: List of ideas (str or dict).
        dog_profile: Profile context for all ideas.
        max_workers: Thread pool size.
        progress_callback: Optional hook to receive ScriptResult as they complete.

    Returns:
        Dict mapping idea_ref → ScriptResult.
    """
    reel_ideas = list(reel_ideas)
    results: Dict[str, ScriptResult] = {}
    if not reel_ideas:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_single_script, idea, dog_profile): idx
            for idx, idea in enumerate(reel_ideas)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results[res.idea_ref] = res
            if progress_callback:
                progress_callback(res)
    return results

# ==================================================
# Script Scoring & Ranking (Heuristic-based)
# ==================================================
def _tokenize_hook(hook_text: str) -> List[str]:
    """Tokenize hook text into normalized word tokens (for diversity checks)."""
    return re.findall(r"[a-zA-Z0-9']+", hook_text.lower())

def compute_script_score(script: Dict[str, Any]) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Compute a weighted score for a script based on:
      - Shot quality (count, detail).
      - Anticipation and pattern interrupts.
      - Hook length and engagement features.
      - Structure completeness and editing cues.
    Returns:
        Total score, component breakdown, and diagnostics.
    """
    comps: Dict[str, float] = {}
    diagnostics: List[str] = []

    shots = script.get("shots", [])
    hashtags = script.get("hashtags", [])
    hook = safe_get(script, "hook.spoken", "") or ""
    editing_notes = script.get("editing_notes","")
    effort = script.get("effort_level","")
    rationale = script.get("rationale","")

    # Shot count score (ideal ~4–5 shots)
    sc = len(shots)
    ideal = 4.5
    comps["shot_count_score"] = max(0, 1 - abs(sc - ideal)/ideal) if sc else 0

    # Shot detail score (ideal action length 25–90 chars)
    if shots:
        avg_len = sum(len(s.get("action","")) for s in shots)/sc
        if avg_len < 15:
            val = avg_len/15 * 0.4
        elif avg_len > 110:
            val = max(0, 1 - (avg_len - 110)/110)
        else:
            val = min(1, (avg_len - 15)/90)
        comps["shot_detail_score"] = max(0, min(val, 1))
    else:
        comps["shot_detail_score"] = 0

    # Anticipation cue score
    anticip_terms = ("anticip", "suspens", "build", "tension", "payoff", "reveal")
    anticip_found = any(
        any(term in (s.get("notes","")+s.get("action","")).lower() for term in anticip_terms)
        for s in shots
    )
    comps["anticipation_score"] = 1.0 if anticip_found else 0.0

    # Pattern interrupt score (zoom, b-roll, jump cuts)
    pi_terms = ("zoom","jump","speed","slow","b-roll","broll","cut","whip","flash")
    pi_found = any(
        any(term in (s.get("notes","")+s.get("action","")).lower() for term in pi_terms)
        for s in shots
    ) or bool(script.get("broll_inserts"))
    comps["pattern_interrupt_score"] = 1.0 if pi_found else 0.0

    # Hook quality (length ≤12 words + engagement markers)
    hook_tokens = _tokenize_hook(hook)
    if hook_tokens:
        length_ok = len(hook_tokens) <= 12
        style_bonus = any(sym in hook.lower() for sym in ("?","!","pov","wait","did"))
        comps["hook_quality_score"] = (0.6 if length_ok else 0.3) + (0.4 if style_bonus else 0)
    else:
        comps["hook_quality_score"] = 0

    # Effort balance (prefer low or medium)
    comps["effort_balance_score"] = {"low":1.0,"medium":0.85,"high":0.55}.get(effort,0.3)

    # Structure completeness (penalize missing keys)
    missing = REQUIRED_TOP_KEYS - set(script.keys())
    comps["structure_completeness"] = 1.0 if not missing else max(0, 1 - len(missing)/len(REQUIRED_TOP_KEYS))

    # Hashtag span (count + diversity)
    hcount = len(hashtags)
    if hcount:
        base = 1.0 if 6 <= hcount <= 10 else max(0, 1 - abs(hcount - 8)/8)
        diversity = len({h.lower() for h in hashtags})/hcount
        comps["hashtag_span_score"] = base * diversity
    else:
        comps["hashtag_span_score"] = 0

    # Rationale quality (length as proxy for thoughtfulness)
    comps["rationale_quality"] = min(1.0, len(rationale)/60) if rationale else 0.0

    # Editing direction cues (presence of edit terms)
    edit_terms = ("cut","zoom","speed","slow","fade","color","grade","transition","jump")
    edit_hits = sum(1 for t in edit_terms if t in editing_notes.lower())
    comps["edit_direction_score"] = min(1.0, edit_hits/3)

    # Weighted sum (normalized to ~1)
    weights = {
        "shot_count_score": 0.10,
        "shot_detail_score": 0.15,
        "anticipation_score": 0.10,
        "pattern_interrupt_score": 0.07,
        "hook_quality_score": 0.15,
        "effort_balance_score": 0.08,
        "structure_completeness": 0.08,
        "hashtag_span_score": 0.07,
        "rationale_quality": 0.05,
        "edit_direction_score": 0.15
    }
    total = sum(comps[k]*w for k, w in weights.items())

    return total, comps, diagnostics

def rank_scripts(results: Dict[str, ScriptResult]) -> List[Dict[str, Any]]:
    """
    Score and rank all successfully generated scripts.
    Adds a small diversity bonus based on uniqueness of hook tokens.
    Returns a sorted leaderboard.
    """
    scored = []
    hooks = []

    # Compute scores per script
    for ref, res in results.items():
        if not res.script_json:
            continue
        score, comps, diag = compute_script_score(res.script_json)
        res.score, res.score_components = score, comps
        hooks.append((_tokenize_hook(safe_get(res.script_json,"hook.spoken","")), ref))
        scored.append(res)

    # Diversity bonus: reward unique hooks
    token_sets = {ref: set(tokens) for tokens, ref in hooks}
    uniqueness_scores = {}
    for ref, toks in token_sets.items():
        overlaps = []
        for other_ref, other_toks in token_sets.items():
            if ref == other_ref:
                continue
            jacc = len(toks & other_toks)/max(1, len(toks | other_toks))
            overlaps.append(jacc)
        avg_overlap = sum(overlaps)/len(overlaps) if overlaps else 0
        uniqueness_scores[ref] = max(0, 1 - avg_overlap)

    # Apply diversity bonus (+0.05 max)
    for res in scored:
        bonus = 0.05 * uniqueness_scores.get(res.idea_ref, 0)
        res.score = (res.score or 0) + bonus
        if res.score_components:
            res.score_components["diversity_bonus"] = bonus

    # Build leaderboard rows
    ranked = sorted(scored, key=lambda r: r.score or 0, reverse=True)
    leaderboard = []
    for i, r in enumerate(ranked, start=1):
        row = {
            "rank": i,
            "idea_ref": r.idea_ref,
            "score": round(r.score or 0, 4),
            **{f"c_{k}": round(v,4) for k,v in (r.score_components or {}).items()},
            "warnings": "; ".join(r.warnings or [])
        }
        leaderboard.append(row)
        if QUALITY_DEBUG:
            print(f"[RANK {i}] {row}")
    return leaderboard

# ==================================================
# Persistence & Reporting
# ==================================================
def save_scripts(results: Dict[str, ScriptResult], out_dir: str = "data/scripts"):
    """
    Save each valid script JSON to the specified directory (filename = idea_ref.json).
    """
    os.makedirs(out_dir, exist_ok=True)
    for ref, res in results.items():
        if res.script_json:
            fname = f"{ref.replace('::','_')}.json"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                json.dump(res.script_json, f, ensure_ascii=False, indent=2)

def scripts_to_text(results: Dict[str, ScriptResult], ranked: List[Dict[str, Any]]) -> str:
    """
    Build a human-readable text report including:
      - Leaderboard summary.
      - Full JSON for each script (sorted by score).
    """
    lines = ["# Ranked Scripts\n", "## Leaderboard"]
    for row in ranked:
        lines.append(
            f"{row['rank']}. {row['idea_ref']} | score={row['score']} | "
            f"hook_quality={row.get('c_hook_quality_score')} | shots={row.get('c_shot_count_score')}"
        )
    lines.append("\n## Scripts Detail\n")
    for r in sorted(results.values(), key=lambda x: x.score or 0, reverse=True):
        lines.append(f"### {r.idea_ref} (score={round(r.score or 0,4)})")
        if r.warnings:
            lines.append(f"Warnings: {r.warnings}")
        if r.script_json:
            lines.append(json.dumps(r.script_json, ensure_ascii=False, indent=2))
        else:
            lines.append(f"[NO SCRIPT] error={r.error}")
        lines.append("")
    return "\n".join(lines)

# ==================================================
# Orchestration Convenience (Generate + Rank)
# ==================================================
def generate_and_rank(
    ideas: Iterable[Union[str, Dict[str, Any]]],
    dog_profile: str,
    save: bool = False,
    out_dir: str = "data/scripts"
) -> Tuple[Dict[str, ScriptResult], List[Dict[str, Any]]]:
    """
    Generate scripts for all ideas, rank them, and optionally save outputs.
    Returns the raw results dict and leaderboard list.
    """
    results = generate_scripts_parallel(ideas, dog_profile)
    leaderboard = rank_scripts(results)
    if save:
        save_scripts(results, out_dir)
        with open(os.path.join(out_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    return results, leaderboard

# ==================================================
# CLI Demo (Sample Execution)
# ==================================================
if __name__ == "__main__":
    # Example ideas (mix of dict and string)
    sample_ideas: List[Union[str, Dict[str, Any]]] = [
        {
            "idea_id": "early-siren-detection",
            "hook": "Did he just hear that?",
            "core_concept": "Mushy senses a distant siren before humans, building anticipation until he howls.",
            "trend_leverage": "Anticipation + reaction meme format",
            "shot_list": ["Freeze ears perk", "Micro tilt", "Launch howl", "Window check"],
            "on_screen_text": ["Wait...", "He hears it", "THE HOWL", "Patrol"],
            "audio_suggestion": "distant-siren-build",
            "caption": "He hears them first.",
            "hashtags": ["#germanshepherd","#howling","#seattledog","#dogsoftiktok","#fyp","#viralreels"]
        },
        "POV: Mushy resisting a mountain of treats until you say 'ball'.",
        {
            "idea_id": "focus-challenge-ball",
            "hook": "Can YOUR dog ignore THIS?",
            "core_concept": "Impulse control ring of treats while Mushy laser-focuses on his ball until release.",
            "trend_leverage": "Challenge / test of discipline pattern",
            "shot_list": ["Set ring","Add distractions","Close-up eyes","Release"],
            "on_screen_text": ["Level 1","Level 2","Focus","GO!"],
            "audio_suggestion": "rising-tension-meme",
            "caption": "Ultimate focus test.",
            "hashtags": ["#dogchallenge","#germanshepherd","#focus","#dogtraining","#seattledog","#viral"]
        }
    ]

    dog_profile = (
        "Mushroom (Musher / Mushy), 4yo German Shepherd in Seattle. Loyal, ball-obsessed, howls at firetrucks, "
        "chases squirrels/rabbits, protective door bark, enjoys homemade treats. Cultural twist: Cantonese mom + American dad."
    )

    results, leaderboard = generate_and_rank(sample_ideas, dog_profile, save=False)
    print(scripts_to_text(results, leaderboard))
