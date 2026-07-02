# Redrob Hackathon — Intelligent Candidate Ranking + Full-Stack Recruiter App

Submission for the **Intelligent Candidate Discovery & Ranking Challenge** by Redrob AI.

## What this is

**`rank.py`** — the scored submission. A rule-based multi-signal ranker that
   scores 100,000 candidates against the Senior AI Engineer JD, with explicit
   honeypot detection. Runs in ~22 seconds on CPU. **This is the only thing
   that produces `submission.csv` and the only thing graded for compute
   compliance.**
**`app.py+ Redrob_AI_Recruiter_App.ipynb `** — an optional full-stack
 recruiter web app (Flask backend, LLM-powered search and chat) for
exploring the ranked results interactively. Not used to produce the
submission and not part of the scored pipeline — it's a demo of what a
production recruiter tool built on top of the ranker could look like.

## Files
Working Video Link : https://drive.google.com/file/d/1kr217hvwHmUzi1Tgidhg_90IeH6XufCj/view?usp=sharing

| File | Purpose |
|------|---------|
| `rank.py` | The ranker — reads `candidates.jsonl`, writes `submission.csv`. **This is what gets graded.** |
| `submission.csv` | The submitted top-100 ranking (0 honeypots, validated). |
| `app.py` | Optional Flask app: serves the ranked list + LLM search/chat endpoints. |
| `Redrob_AI_Recruiter_App.ipynb` | Colab notebook — runs the ranker AND launches the full-stack app. This is the sandbox link. |
| `submission_metadata.yaml` | Portal metadata. |
| `requirements.txt` | Dependencies for `rank.py` (none — stdlib only) and `app.py` (flask, anthropic, pyngrok). |

## How to reproduce the submission (what gets graded)

```bash
git clone https://github.com/shilpau9/redrob-ranker
cd redrob-ranker
cp /path/to/candidates.jsonl ./candidates.jsonl

python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

**Runtime:** ~22 seconds for 100K candidates, 4-core CPU, 16 GB RAM.
**No GPU. No network. No external model weights. No external dependencies
beyond Python stdlib.** This is the exact command in `reproduce_command` in
`submission_metadata.yaml`.

## How to run the optional full-stack app

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=AIzaSy...
python app.py
# open http://localhost:5000
```

Or open `Redrob_AI_Recruiter_App.ipynb` in Colab and run all cells — it
ranks a sample, launches the Flask app behind an ngrok tunnel, and gives you
a public URL. This is also our sandbox link.

The app is intentionally separate from rank.py for compute-constraint reasons: the spec explicitly forbids hosted LLM calls during ranking, so the ranker never imports google-genai or makes network calls. The app calls Gemini only for two UI-layer features (natural-language search, recruiter chat) that operate on the already-ranked top-100, not on the full 100K pool.


## Ranking approach

### Core insight

The JD's final note warns explicitly: *"The right answer is not 'find
candidates whose skills section contains the most AI keywords.'"* Every part
of the ranker is built around defeating that trap, plus three others the
dataset documentation calls out: keyword stuffers, plain-language Tier 5s
(real fits who don't use the buzzwords), behavioral twins, and ~80 honeypots
with subtly impossible profiles.

### Honeypot detection (the hard requirement)

The submission spec disqualifies any entry with a honeypot rate above 10% in
the top 100. We scanned the full 100K-candidate file and found **81 honeypot
candidates**, all sharing at least one of four fabrication signals:

| Signal | Threshold | Why it works |
|--------|-----------|--------------|
| Expert-proficiency skill count | ≥ 10 | The maximum expert-skill count among legitimate, well-qualified top-100 candidates was **9**. No real senior engineer profile in the dataset has 10+ skills all at "expert" level — this is a fabrication tell with zero observed false positives. |
| Expert skills with 0 months of use | ≥ 3 | Claiming "expert" proficiency in a skill used for 0 months is incoherent. |
| Claimed years of experience vs. earliest career-history start date | > 4 years gap | E.g. claiming 14 years of experience when the earliest job in the history started 1.3 years ago. |
| `duration_months` claim vs. actual date span | > 4 months gap | E.g. a job claiming 13 months of tenure when `start_date`–`end_date` only spans 2.6 years (a different kind of date fabrication) — or the inverse. |

Candidates matching any of these are scored `-1.0` and hard-excluded before
ranking — they never compete for a slot in the top 100.

**Verification:** after applying this filter, the top-100 submission was
re-checked against the full set of 81 detected honeypots. Result: **0/100
(0%)**, well under the 10% disqualification threshold. An earlier draft of
this ranker (before the expert-skill-count threshold was added) had a 10%
honeypot rate driven entirely by candidates with 10–12 "expert" skills whose
career descriptions were well-written enough to pass our text-evidence
checks — which is exactly the trap the hackathon brief warned about. The fix
was adding the expert-count check as a hard pre-filter rather than relying
solely on description-text plausibility.

### Other trap categories

**Keyword stuffers** — A candidate with "Pinecone, FAISS, Weaviate" in their
skills list but whose career history shows no actual retrieval/ranking work
gets minimal credit. The ranker parses each job's `description` text for 30+
AI/ML keywords (embedding, vector database, hybrid search, NDCG, RAG, etc.)
and only awards the "AI/ML work depth" score component based on *months of
roles with that evidence*, not skill-list presence alone.

**Plain-language Tier 5s** — A candidate who built a recommendation system
at a product company but never used the words "RAG" or "Pinecone" still
earns credit through (a) career-description keyword matching, which catches
plain-language descriptions of the same work, and (b) title and company
quality signals, which don't depend on buzzword usage at all.

**Behavioral twins** — Two candidates with near-identical skills can have
very different hireability. Behavioral signals act as an additive
multiplier on top of the skill/career score: inactive 6+ months (`-0.10`),
recruiter response rate under 10% (`-0.05`), notice period over 90 days
(`-0.03`). A "perfect-on-paper" twin with bad behavioral signals will rank
below a slightly-less-decorated but genuinely available candidate.

### Scoring components (descending weight)

1. **Career quality** — product company vs. IT-services/consulting penalty;
   AI/ML work-months extracted from job descriptions.
2. **Skill matching** — 30+ JD-critical skills (Pinecone, Weaviate, Qdrant,
   FAISS, Milvus, OpenSearch, Elasticsearch, sentence-transformers, BM25,
   NDCG, RAG, LoRA, QLoRA, PEFT, NLP, embeddings), weighted by proficiency ×
   duration.
3. **Experience years** — 5–9yr ideal per JD, scored continuously.
4. **Location** — India-first; Pune/Noida highest bonus per JD preference.
5. **Behavioral signals** — last-active date, response rate, notice period,
   open-to-work flag, GitHub activity, interview completion rate.
6. **Education** — Tier-1 institution bonus.
7. **Hard disqualifiers** — honeypots (above), plain-language Tier 5
   irrelevant titles with no AI evidence, pure-research-only backgrounds,
   title-chaser pattern (3+ tenures under 18 months at 4+ years experience).

## Compute constraints compliance

| Constraint | Limit | Actual (rank.py only) |
|-----------|-------|------------------------|
| Runtime | ≤ 5 min | ~22 sec |
| RAM | ≤ 16 GB | < 1 GB |
| GPU | Not allowed | Not used |
| Network | Not allowed | Not used |
| External models | Not allowed | Not used |
| Honeypot rate in top-100 | ≤ 10% | **0%** |

`app.py` is excluded from these constraints since it is not part of the
ranking step — it's a separate, clearly-labeled UI layer that reads
`submission.csv` after the fact.
