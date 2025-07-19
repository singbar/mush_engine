"""
script_writer_workflow.py

Generate detailed, execution-ready short-form dog reel scripts from idea inputs,
then rank them via heuristic quality scoring.

Features:
- Accept raw string or structured dict ideas.
- Long-context capable Chat model (OpenAI).
- Parallel generation with retries & exponential backoff.
- JSON schema style validation & optional repair reprompt.
- Heuristic ranking & leaderboard output.
- CLI demonstration.

Set ENV:
  OPENAI_API_KEY
Optional ENV:
  SCRIPT_MODEL_NAME (default gpt-4o)
  SCRIPT_MAX_WORKERS (default 6)

Author: (Your project)
"""

from __future__ import annotations
import os
import time
import math
import json
import uuid
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== LLM Import ==========
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI  # fallback

from langchain.schema import HumanMessage, SystemMessage

# ========== Configuration ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("SCRIPT_MODEL_NAME", "gpt-4o")
MAX_WORKERS = int(os.getenv("SCRIPT_MAX_WORKERS", "6"))
TEMPERATURE = 0.65
MAX_MODEL_TOKENS = 900
RETRY_ATTEMPTS = 4
BASE_BACKOFF = 2.0
REPAIR_ATTEMPTS = 1            # number of repair re-prompts if validation fails
QUALITY_DEBUG = False          # set True to print scoring internals per idea

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")

# ========== System & Prompt Templates ==========

SCRIPT_SYSTEM = (
    "You are a senior short-form video director for high-retention pet Reels/TikToks. "
    "You output ONLY valid JSON meeting the requested schema."
)

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

SCRIPT_USER_TEMPLATE = """Dog Profile:
{dog_profile}

Original Idea (JSON or text):
{idea_json}

TASK:
Transform the idea into an execution-ready script (do not simply restate the idea).
Elevate specificity: pacing, camera, anticipation, pattern interrupts, editing direction.

OUTPUT RULES:
Return ONLY one JSON object with schema (see below). No markdown, no commentary.
{schema_block}

Constraints:
- 3–6 primary shots (condense if verbose).
- Hook spoken & on_screen may differ for contrast; spoken <12 words.
- Provide durations in seconds (approx; float acceptable).
- Each action must be concrete (avoid generic 'dog does something').
- Include at least one anticipation/suspense or payoff moment.
- Provide at least one pattern interrupt (zoom, b-roll insert, jump cut, etc.).
- hashtags: 6–10, mix broad (#dogsoftiktok), breed (#germanshepherd), locale (#seattledog), trend/meme.
- Avoid emojis; no trailing period on hashtags.
- rationale <=180 chars.
- editing_notes should reference at least 1 editing technique (cut/zoom/speed/color/sound design).
- JSON must be strictly valid; no trailing commas; keys spelled exactly as required.
"""

# ========== Data Structures ==========

@dataclass
class ScriptResult:
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

# ========== LLM Instance ==========
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_MODEL_TOKENS
)

# ========== Helpers ==========

def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)

def truncate(text: str, max_tokens: int, margin: int = 200) -> str:
    if estimate_tokens(text) + margin <= max_tokens:
        return text
    target_chars = (max_tokens - margin) * 4
    return text[:target_chars] + "... [TRIMMED]"

def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ========== Prompt Builder ==========

def build_script_prompt(idea: Union[str, Dict[str, Any]], dog_profile: str) -> Tuple[List[Any], str]:
    if isinstance(idea, dict):
        compact_keys = {
            "idea_id","hook","core_concept","trend_leverage",
            "shot_list","on_screen_text","audio_suggestion","caption","hashtags"
        }
        compact = {k: idea[k] for k in idea if k in compact_keys}
        if "idea_id" not in compact:
            compact["idea_id"] = idea.get("idea_id") or f"idea-{uuid.uuid4().hex[:6]}"
        idea_json = json.dumps(compact, ensure_ascii=False)
        idea_ref = compact["idea_id"]
        source_type = "dict"
    else:
        idea_json = idea.strip()
        idea_ref = f"raw-{uuid.uuid4().hex[:6]}"
        source_type = "string"

    idea_json = truncate(idea_json, 3000)
    user = SCRIPT_USER_TEMPLATE.format(
        dog_profile=dog_profile.strip(),
        idea_json=idea_json,
        schema_block=SCHEMA_BLOCK.strip()
    )
    messages = [
        SystemMessage(content=SCRIPT_SYSTEM),
        HumanMessage(content=user)
    ]
    return messages, idea_ref + f"::{source_type}"

# ========== JSON Validation & Repair ==========

REQUIRED_TOP_KEYS = {
    "idea_id","summary","hook","shots","audio_direction",
    "editing_notes","cta","caption","hashtags","effort_level",
    "gear_checklist","publishing_tips","rationale","generated_at"
}

def validate_script(script: Dict[str, Any]) -> List[str]:
    warnings = []
    missing = REQUIRED_TOP_KEYS - set(script.keys())
    if missing:
        warnings.append(f"Missing keys: {sorted(missing)}")
    # hook subkeys
    hook = script.get("hook", {})
    if not isinstance(hook, dict) or any(k not in hook for k in ("spoken","on_screen")):
        warnings.append("hook must include spoken & on_screen")
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
    hashtags = script.get("hashtags", [])
    if not (6 <= len(hashtags) <= 10):
        warnings.append("hashtags count not 6–10")
    rationale = script.get("rationale","")
    if len(rationale) > 180:
        warnings.append("rationale >180 chars")
    effort = script.get("effort_level","")
    if effort not in {"low","medium","high"}:
        warnings.append("effort_level invalid")
    return warnings

def build_repair_prompt(original_raw: str, warnings: List[str]) -> List[Any]:
    repair_instructions = (
        "The previous JSON did not meet validation. Fix ONLY the issues listed below; "
        "return a SINGLE corrected JSON object, retaining good content, adjusting minimal necessary fields.\n"
        f"Issues: {warnings}\n"
        "Return ONLY valid JSON. No commentary."
    )
    return [
        SystemMessage(content=SCRIPT_SYSTEM),
        HumanMessage(content=repair_instructions),
        HumanMessage(content=original_raw)
    ]

# ========== Core Single Generation ==========

def generate_single_script(
    idea: Union[str, Dict[str, Any]],
    dog_profile: str,
    repair_attempts: int = REPAIR_ATTEMPTS
) -> ScriptResult:
    messages, idea_ref = build_script_prompt(idea, dog_profile)
    start = time.time()
    attempts = 0
    raw_text = None
    script_json = None
    error = None
    repaired = False

    for attempt in range(RETRY_ATTEMPTS):
        attempts = attempt + 1
        try:
            resp = llm.invoke(messages)
            raw_text = resp.content.strip()
            # salvage JSON
            if not raw_text.startswith("{"):
                if "{" in raw_text and "}" in raw_text:
                    first = raw_text.index("{")
                    last = raw_text.rindex("}") + 1
                    raw_text_candidate = raw_text[first:last]
                    raw_text = raw_text_candidate
            script_json = json.loads(raw_text)
            if isinstance(script_json, dict):
                script_json.setdefault("generated_at", iso_now())
                break
            else:
                error = "Top level JSON not object"
        except Exception as e:
            error = f"Gen attempt {attempt+1} error: {e}"
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(BASE_BACKOFF * (2 ** attempt))
            else:
                break

    warnings = []
    if script_json:
        warnings = validate_script(script_json)

    # Attempt repair if needed
    if warnings and repair_attempts > 0:
        repaired = True
        repair_msgs = build_repair_prompt(raw_text or "{}", warnings)
        try:
            resp = llm.invoke(repair_msgs)
            repaired_raw = resp.content.strip()
            if not repaired_raw.startswith("{"):
                if "{" in repaired_raw and "}" in repaired_raw:
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

# ========== Parallel Generation ==========

def generate_scripts_parallel(
    reel_ideas: Iterable[Union[str, Dict[str, Any]]],
    dog_profile: str,
    max_workers: int = MAX_WORKERS,
    progress_callback: Optional[Callable[[ScriptResult], None]] = None
) -> Dict[str, ScriptResult]:
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

# ========== Ranking Heuristics ==========

def _tokenize_hook(hook_text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", hook_text.lower())

def compute_script_score(script: Dict[str, Any]) -> Tuple[float, Dict[str, float], List[str]]:
    comps: Dict[str, float] = {}
    diagnostics: List[str] = []
    shots = script.get("shots", [])
    hashtags = script.get("hashtags", [])
    hook = safe_get(script, "hook.spoken", "") or ""
    editing_notes = script.get("editing_notes","")
    effort = script.get("effort_level","")
    rationale = script.get("rationale","")

    # shot_count_score (ideal = 4 or 5)
    sc = len(shots)
    if sc == 0:
        comps["shot_count_score"] = 0
    else:
        ideal = 4.5
        comps["shot_count_score"] = max(0, 1 - abs(sc - ideal)/ideal)  # in [0,1]

    # shot_detail_score (avg action length 25–90 chars sweet spot)
    if shots:
        avg_len = sum(len(s.get("action","")) for s in shots)/sc
        if avg_len < 15:
            val = avg_len/15 * 0.4
        elif avg_len > 110:
            val = max(0, 1 - (avg_len - 110)/110)
        else:
            val = min(1, (avg_len - 15)/(90))  # scale into region
        comps["shot_detail_score"] = max(0, min(val, 1))
    else:
        comps["shot_detail_score"] = 0

    # anticipation_score (look for patterns)
    anticipation_terms = ("anticip", "suspens", "build", "tension", "payoff", "reveal")
    anticip_found = any(
        any(term in (s.get("notes","")+s.get("action","")).lower() for term in anticipation_terms)
        for s in shots
    )
    comps["anticipation_score"] = 1.0 if anticip_found else 0.0

    # pattern_interrupt_score (zoom, cut, broll, speed)
    pi_terms = ("zoom","jump","speed","slow","b-roll","broll","cut","whip","flash")
    pi_found = any(
        any(term in (s.get("notes","")+s.get("action","")).lower() for term in pi_terms)
        for s in shots
    ) or bool(script.get("broll_inserts"))
    comps["pattern_interrupt_score"] = 1.0 if pi_found else 0.0

    # hook_quality_score (<=12 words, presence of question/POV/surprise)
    hook_tokens = _tokenize_hook(hook)
    if not hook_tokens:
        comps["hook_quality_score"] = 0
    else:
        length_ok = len(hook_tokens) <= 12
        style_bonus = any(sym in hook.lower() for sym in ("?","!","pov","wait","did"))
        comps["hook_quality_score"] = (0.6 if length_ok else 0.3) + (0.4 if style_bonus else 0)

    # effort_balance_score prefer low or medium
    if effort == "low":
        comps["effort_balance_score"] = 1.0
    elif effort == "medium":
        comps["effort_balance_score"] = 0.85
    elif effort == "high":
        comps["effort_balance_score"] = 0.55
    else:
        comps["effort_balance_score"] = 0.3

    # structure_completeness
    missing = REQUIRED_TOP_KEYS - set(script.keys())
    comps["structure_completeness"] = 1.0 if not missing else max(0, 1 - len(missing)/len(REQUIRED_TOP_KEYS))

    # hashtag_span_score
    hcount = len(hashtags)
    if hcount == 0:
        comps["hashtag_span_score"] = 0
    else:
        base = 1.0 if 6 <= hcount <= 10 else max(0, 1 - abs(hcount - 8)/8)
        diversity = len({h.lower() for h in hashtags})/hcount
        comps["hashtag_span_score"] = base * diversity

    # rationale_quality
    comps["rationale_quality"] = min(1.0, len(rationale)/60) if rationale else 0.0

    # edit_direction_score (look for edit cues)
    edit_terms = ("cut","zoom","speed","slow","fade","color","grade","transition","jump")
    edit_hits = sum(1 for t in edit_terms if t in editing_notes.lower())
    comps["edit_direction_score"] = min(1.0, edit_hits/3)

    # Weighted sum
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
    # First pass scoring (without diversity bonus)
    scored = []
    hooks = []
    for ref, res in results.items():
        if not res.script_json:
            continue
        score, comps, diag = compute_script_score(res.script_json)
        res.score = score
        res.score_components = comps
        hooks.append((_tokenize_hook(safe_get(res.script_json,"hook.spoken","")), ref))
        scored.append(res)

    # Diversity bonus across hooks: reward unique token sets
    token_sets = {ref: set(tokens) for tokens, ref in hooks}
    uniqueness_scores = {}
    for ref, toks in token_sets.items():
        overlap_counts = []
        for other_ref, other_toks in token_sets.items():
            if ref == other_ref: 
                continue
            jacc = len(toks & other_toks)/max(1, len(toks | other_toks))
            overlap_counts.append(jacc)
        avg_overlap = sum(overlap_counts)/len(overlap_counts) if overlap_counts else 0
        uniqueness_scores[ref] = max(0, 1 - avg_overlap)  # higher is better

    # Apply diversity bonus (up to +0.05)
    for res in scored:
        div_bonus = 0.05 * uniqueness_scores.get(res.idea_ref, 0)
        res.score = (res.score or 0) + div_bonus
        if res.score_components:
            res.score_components["diversity_bonus"] = div_bonus

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

# ========== Persistence & Display ==========

def save_scripts(results: Dict[str, ScriptResult], out_dir: str = "data/scripts"):
    os.makedirs(out_dir, exist_ok=True)
    for ref, res in results.items():
        if res.script_json:
            fname = f"{ref.replace('::','_')}.json"
            path = os.path.join(out_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res.script_json, f, ensure_ascii=False, indent=2)

def scripts_to_text(results: Dict[str, ScriptResult], ranked: List[Dict[str, Any]]) -> str:
    lines = ["# Ranked Scripts\n"]
    # leaderboard
    lines.append("## Leaderboard")
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

# ========== Orchestrator Convenience ==========

def generate_and_rank(
    ideas: Iterable[Union[str, Dict[str, Any]]],
    dog_profile: str,
    save: bool = False,
    out_dir: str = "data/scripts"
) -> Tuple[Dict[str, ScriptResult], List[Dict[str, Any]]]:
    results = generate_scripts_parallel(ideas, dog_profile)
    leaderboard = rank_scripts(results)
    if save:
        save_scripts(results, out_dir=out_dir)
        # Save leaderboard
        with open(os.path.join(out_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    return results, leaderboard

# ========== CLI Demo ==========

if __name__ == "__main__":
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
    text = scripts_to_text(results, leaderboard)
    print(text)
