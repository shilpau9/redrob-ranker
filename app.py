#!/usr/bin/env python3
"""
Redrob AI Recruiter — Full-Stack Web App
==========================================
Optional UI layer on top of rank.py's output. NOT used to produce the
submission CSV and NOT subject to the ranking compute constraints (no GPU /
no network rules apply only to the ranking step itself, per submission_spec
Section 3).

Serves:
  GET  /                  -> recruiter dashboard (rank, search, chat, trap inspector)
  GET  /api/candidates    -> filtered ranked candidate list (reads submission.csv + candidates.jsonl)
  POST /api/search        -> natural-language candidate search via Gemini
  POST /api/chat          -> streaming recruiter chat assistant via Gemini
  GET  /api/honeypots     -> honeypot detection stats (for the Trap Inspector tab)

Run:
    pip install google-genai pydantic flask
    export GEMINI_API_KEY=AIzaSy...
    python app.py
    # open http://localhost:5000
"""

import os
import csv
import json
from typing import List

from flask import Flask, jsonify, request, Response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from rank import is_honeypot, score_candidate, CURRENT_DATE 

app = Flask(__name__)

# Initialize the standard Google GenAI client
# It automatically picks up GEMINI_API_KEY from the environment variables
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

CANDIDATES_PATH = os.environ.get("CANDIDATES_PATH", "candidates.jsonl")
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "submission.csv")

# ─── Pydantic Schemas for Structured JSON Search ──────────────────────────────

class CandidateMatch(BaseModel):
    id: str
    matchScore: float = Field(description="Match score between 0.0 and 1.0")
    explanation: str = Field(description="2 sentences max, specific to why they match the query")

class SearchResponseSchema(BaseModel):
    matches: List[CandidateMatch]


# ─── Load data once at startup ───────────────────────────────────────────────

def load_ranked_candidates():
    """Join submission.csv (rank/score/reasoning) with full candidate records."""
    if not os.path.exists(SUBMISSION_PATH):
        return []

    ranked_meta = {}
    with open(SUBMISSION_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranked_meta[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"],
            }

    if not os.path.exists(CANDIDATES_PATH):
        return [
            {"candidate_id": cid, **meta}
            for cid, meta in sorted(ranked_meta.items(), key=lambda x: x[1]["rank"])
        ]

    results = []
    needed = set(ranked_meta)
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
            except Exception:
                continue
            cid = c.get("candidate_id")
            if cid in needed:
                results.append({
                    "candidate_id": cid,
                    **ranked_meta[cid],
                    "profile": c.get("profile", {}),
                    "skills": c.get("skills", [])[:10],
                    "career_history": c.get("career_history", [])[:3],
                    "education": c.get("education", [])[:1],
                    "redrob_signals": c.get("redrob_signals", {}),
                })
                needed.discard(cid)
            if not needed:
                break

    results.sort(key=lambda x: x["rank"])
    return results


def scan_honeypot_stats():
    """Run honeypot detection over the full pool for the Trap Inspector tab."""
    if not os.path.exists(CANDIDATES_PATH):
        return {"total_scanned": 0, "honeypots_found": 0, "honeypots_in_top100": 0}

    honeypot_ids = set()
    total = 0
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                c = json.loads(line)
            except Exception:
                continue
            total += 1
            if is_honeypot(c):
                honeypot_ids.add(c["candidate_id"])

    top100_ids = set()
    if os.path.exists(SUBMISSION_PATH):
        with open(SUBMISSION_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                top100_ids.add(row["candidate_id"])

    in_top100 = honeypot_ids & top100_ids
    return {
        "total_scanned": total,
        "honeypots_found": len(honeypot_ids),
        "honeypots_in_top100": len(in_top100),
        "honeypot_rate_pct": round(len(in_top100) / max(1, len(top100_ids)) * 100, 1),
    }


RANKED_CANDIDATES = load_ranked_candidates()
HONEYPOT_STATS = scan_honeypot_stats()

print(f"Loaded {len(RANKED_CANDIDATES)} ranked candidates")
print(f"Honeypot scan: {HONEYPOT_STATS}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(here, "app_frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>app_frontend.html not found</h1><p>See README for setup.</p>"


@app.route("/api/candidates")
def get_candidates():
    min_yoe = float(request.args.get("min_yoe", 0))
    min_score = float(request.args.get("min_score", 0))
    loc = request.args.get("loc", "").lower()
    avail = request.args.get("avail", "")

    filtered = []
    for c in RANKED_CANDIDATES:
        profile = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        if profile.get("years_of_experience", 0) < min_yoe:
            continue
        if c["score"] < min_score:
            continue
        if loc and loc not in profile.get("location", "").lower():
            continue
        if avail == "open" and not sig.get("open_to_work_flag"):
            continue
        if avail == "imm" and sig.get("notice_period_days", 999) > 15:
            continue
        if avail == "30" and sig.get("notice_period_days", 999) > 30:
            continue
        filtered.append(c)

    return jsonify(filtered)


@app.route("/api/honeypots")
def get_honeypot_stats():
    return jsonify(HONEYPOT_STATS)


@app.route("/api/search", methods=["POST"])
def search():
    q = request.json.get("query", "")
    pool = RANKED_CANDIDATES[:50]
    pool_summary = json.dumps([
        {
            "id": c["candidate_id"],
            "name": c.get("profile", {}).get("anonymized_name"),
            "title": c.get("profile", {}).get("current_title"),
            "company": c.get("profile", {}).get("current_company"),
            "location": c.get("profile", {}).get("location"),
            "yoe": c.get("profile", {}).get("years_of_experience"),
            "skills": [s["name"] for s in c.get("skills", [])[:5]],
            "open": c.get("redrob_signals", {}).get("open_to_work_flag"),
            "notice": c.get("redrob_signals", {}).get("notice_period_days"),
            "github": c.get("redrob_signals", {}).get("github_activity_score"),
            "score": c["score"],
        }
        for c in pool
    ], default=str)[:10000]

    # Leverage Gemini's native Structured Outputs for reliability
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=(
            f'Recruiter query: "{q}"\n'
            f"Candidates:\n{pool_summary}\n"
            "Return up to 5 best matches strictly following the schema layout."
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SearchResponseSchema,
            temperature=0.1
        )
    )

    # Parse JSON structured result safely
    response_data = json.loads(response.text)
    matches = response_data.get("matches", [])

    full = []
    for m in matches:
        c = next((x for x in pool if x["candidate_id"] == m["id"]), None)
        if c:
            full.append({**c, "matchScore": m["matchScore"], "explanation": m["explanation"]})
    return jsonify(full)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    history = data.get("history", [])

    top10 = RANKED_CANDIDATES[:10]
    system_instruction = (
        "You are a senior AI recruiter assistant at Redrob AI, helping a recruiter "
        f"hire a Senior AI Engineer. {HONEYPOT_STATS['honeypots_found']} honeypot "
        "candidates were detected and excluded before ranking (impossible profiles: "
        "10+ expert skills, fabricated dates, etc). You have these ranked candidates:\n"
        + json.dumps([
            {
                "rank": c["rank"], "score": c["score"], "id": c["candidate_id"],
                "name": c.get("profile", {}).get("anonymized_name"),
                "title": c.get("profile", {}).get("current_title"),
                "company": c.get("profile", {}).get("current_company"),
                "location": c.get("profile", {}).get("location"),
                "yoe": c.get("profile", {}).get("years_of_experience"),
                "skills": [s["name"] for s in c.get("skills", [])[:6]],
                "reasoning": c.get("reasoning"),
                "open": c.get("redrob_signals", {}).get("open_to_work_flag"),
                "notice": c.get("redrob_signals", {}).get("notice_period_days"),
            }
            for c in top10
        ], default=str)
        + "\nBe concise and specific. Use candidate names."
    )

    # Convert chat history array format to Gemini Content Types array format
    gemini_contents = []
    for turn in history:
        # Translate roles: 'assistant' -> 'model'
        role = "model" if turn.get("role") == "assistant" else "user"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn.get("content", ""))]
            )
        )

    def generate():
        response_stream = client.models.generate_content_stream(
            model=MODEL_ID,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1000
            )
        )
        for chunk in response_stream:
            if chunk.text:
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)