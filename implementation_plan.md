# Redrob AI Candidate Ranking System — v15 FINAL

Fully self-contained. Every function defined. No cross-references. Ready to code.

---

## 1. Architecture

```
PRE-COMPUTATION (Colab T4 GPU, ~22 min)
  ├── Save bi-encoder + cross-encoder locally
  ├── Normalize + encode 100K candidates → float16
  ├── Build FAISS IndexFlatIP
  ├── Deduplicate + normalize + encode ~10K skill names
  ├── Pre-compute skill matches → pickle dict
  ├── Pre-compute archetype max scores → npy
  ├── Save candidate_order.npy
  ├── Metadata → parquet (flat) + pickle (nested)

RANKING (≤5 min, CPU, 16GB, no network)
  ├── Load + assert alignment + cast float32 + vectorize semantics
  ├── STAGE 1: FAISS Union → ~3500 → hard-exclude → top 500
  ├── STAGE 2: Cross-encoder on 500 (sigmoid normalized)
  ├── STAGE 3: 7-Feature scoring → top 100
  └── STAGE 4: Data-driven reasoning → CSV
```

---

## 2. Pre-Computation

### 2.1 Models

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

bi_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
bi_model.save('artifacts/models/bi/')

ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
ce_model.save('artifacts/models/ce/')
```

### 2.2 JD Queries (5)

```python
JD_QUERIES = [
    "search ranking recommendation retrieval systems deployed production users matching relevance discovery personalization",
    "embeddings vector database FAISS Milvus Pinecone Weaviate Qdrant hybrid search dense retrieval infrastructure",
    "shipped production deployed real users scale serving latency real-time traffic API microservice",
    "python pytorch machine learning engineering evaluation NDCG MRR A/B testing code quality",
    "LLM fine-tuning LoRA transformers learning to rank XGBoost distributed systems",
]
jd_query_embeds = bi_model.encode(JD_QUERIES, normalize_embeddings=True)
np.save("artifacts/jd_query_embeddings.npy", jd_query_embeds.astype(np.float32))
```

### 2.3 JD Skill Requirements (20)

```python
JD_SKILLS = [
    # MUST-HAVE (0-9)
    "embeddings-based retrieval and semantic search",
    "vector databases and approximate nearest neighbor search",
    "natural language processing and text understanding",
    "ranking systems and relevance scoring",
    "search systems and information retrieval",
    "recommendation systems and collaborative filtering",
    "Python production-quality code",
    "PyTorch or TensorFlow deep learning frameworks",
    "evaluation frameworks NDCG MRR MAP precision recall",
    "machine learning model deployment and serving",
    # NICE-TO-HAVE (10-19)
    "LLM fine-tuning LoRA QLoRA PEFT",
    "learning to rank XGBoost LightGBM gradient boosting",
    "distributed systems and large-scale data processing",
    "Docker Kubernetes containerized ML deployment",
    "MLOps MLflow Airflow experiment tracking",
    "Elasticsearch OpenSearch Solr search infrastructure",
    "data pipelines ETL feature engineering",
    "cloud platforms AWS GCP Azure SageMaker",
    "API design REST FastAPI microservices",
    "HR-tech recruiting technology talent marketplace",
]
jd_skill_embeds = bi_model.encode(JD_SKILLS, normalize_embeddings=True)
np.save("artifacts/jd_skill_embeddings.npy", jd_skill_embeds.astype(np.float32))
```

### 2.4 Archetypes (10)

```python
ARCHETYPES = [
    "Search engineer who built and deployed search systems with ranking and relevance optimization",
    "Recommendation systems engineer who built collaborative filtering and content-based recommendation engines",
    "Retrieval engineer working on dense retrieval, semantic search, and embedding-based document matching",
    "Matching engineer who built candidate-job matching or marketplace matching systems using ML",
    "Ranking engineer who implemented learning-to-rank models and evaluation frameworks",
    "NLP engineer who built information retrieval and text understanding systems for production use",
    "Talent discovery engineer who built systems to find and rank candidates for recruiting platforms",
    "ML platform engineer who built and maintained embedding pipelines and vector search infrastructure",
    "Applied scientist who shipped ranking models and recommendation algorithms to real users",
    "Data scientist focused on search relevance, query understanding, and retrieval quality metrics",
]
arch_embeds = bi_model.encode(ARCHETYPES, normalize_embeddings=True)
np.save("artifacts/archetype_embeddings.npy", arch_embeds.astype(np.float32))
```

### 2.5 Candidate Embeddings (pre-normalized float16)

```python
import json, re

ordered_candidates = []
all_embeddings = []

for line in open("candidates.jsonl"):
    candidate = json.loads(line)
    ordered_candidates.append(candidate)

    # STRUCTURED TEXT — section labels improve embedding quality
    # Skills listed first and labeled get proper emphasis in 384-d space
    skills_text = ", ".join(s.get("name", "") for s in candidate.get("skills", []))
    career_text = " ".join(
        (r.get("description", "") or "") for r in candidate.get("career_history", [])
    )
    summary_text = candidate.get("profile", {}).get("summary", "") or ""

    profile_text = (
        f"Skills: {skills_text}. "
        f"Experience: {career_text} "
        f"Summary: {summary_text}"
    )
    emb = bi_model.encode(profile_text, normalize_embeddings=True)
    all_embeddings.append(emb)

embeddings = np.array(all_embeddings, dtype=np.float16)
np.save("artifacts/candidate_embeddings.npy", embeddings)

candidate_ids = np.array([c["candidate_id"] for c in ordered_candidates])
np.save("artifacts/candidate_order.npy", candidate_ids)
```

### 2.6 FAISS Index

```python
import faiss
emb_f32 = embeddings.astype(np.float32)
index = faiss.IndexFlatIP(384)
index.add(emb_f32)
faiss.write_index(index, "artifacts/candidate.index")
```

### 2.7 Archetype Max Scores

```python
# VECTORIZED — single matmul instead of 100K Python loop
archetype_max = np.max(emb_f32 @ arch_embeds.T, axis=1).astype(np.float16)
np.save("artifacts/archetype_max_scores.npy", archetype_max)
```

### 2.8 Skill Normalization + Deduplication + Matching

```python
def normalize_skill_name(name):
    name = name.lower().strip()
    name = re.sub(r'[-_/]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    aliases = {
        "torch": "pytorch", "py torch": "pytorch",
        "tf": "tensorflow", "tensor flow": "tensorflow",
        "scikit learn": "sklearn", "sci kit learn": "sklearn",
        "elastic search": "elasticsearch",
        "open search": "opensearch",
        "amazon web services": "aws",
        "google cloud platform": "gcp", "google cloud": "gcp",
        "microsoft azure": "azure",
        "hf transformers": "transformers",
        "huggingface": "transformers", "hugging face": "transformers",
    }
    return aliases.get(name, name)

# Collect unique normalized names
all_names = set()
for c in ordered_candidates:
    for s in c.get("skills", []):
        all_names.add(normalize_skill_name(s.get("name", "")))
unique_names = sorted(all_names - {""})

# Encode unique names ONCE
unique_embeds = bi_model.encode(unique_names, batch_size=256,
                                normalize_embeddings=True)
skill_embed_map = dict(zip(unique_names, unique_embeds))

# Build skill_lookup: candidate_id → list of match dicts
import pickle
from sklearn.metrics.pairwise import cosine_similarity

skill_lookup = {}
for c in ordered_candidates:
    cid = c["candidate_id"]
    matches = []
    for s in c.get("skills", []):
        norm = normalize_skill_name(s.get("name", ""))
        if norm not in skill_embed_map:
            continue
        se = skill_embed_map[norm]
        sims = se @ jd_skill_embeds.T  # (20,) dot = cosine
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        matches.append({
            "skill_name": s.get("name", ""),
            "norm_name": norm,
            "best_jd_match_score": best_score,
            "best_jd_req_idx": best_idx,
            "proficiency": s.get("proficiency", "intermediate"),
            "endorsements": s.get("endorsements", 0),
            "duration_months": s.get("duration_months", 0),
        })
    skill_lookup[cid] = matches

pickle.dump(skill_lookup, open("artifacts/skill_matches.pkl", "wb"))
```

### 2.9 Skill Threshold Calibration Test

Run this during precompute to verify the 0.50 threshold is appropriate:

```python
# --- CALIBRATION — verify threshold before committing ---
test_skills = [
    "pytorch", "faiss", "elasticsearch", "embeddings",
    "recommendation systems", "python", "docker", "kubernetes",
    "natural language processing", "transformers",
]
test_embeds = bi_model.encode(test_skills, normalize_embeddings=True)
print("\n=== SKILL THRESHOLD CALIBRATION ===")
for name, emb in zip(test_skills, test_embeds):
    sims = emb @ jd_skill_embeds.T
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    print(f"  {name:30s} -> JD[{best_idx:2d}] = {best_score:.3f}  "
          f"({'PASS' if best_score > 0.50 else 'BELOW THRESHOLD'})")

# If any obviously relevant skill scores below 0.50, lower threshold to 0.45
# Update SKILL_MATCH_THRESHOLD accordingly
SKILL_MATCH_THRESHOLD = 0.50  # adjust based on calibration output
np.save("artifacts/skill_threshold.npy", np.array([SKILL_MATCH_THRESHOLD]))
print(f"\nUsing threshold: {SKILL_MATCH_THRESHOLD}")
```

### 2.10 Metadata Extraction

**Parquet (flat):**
```python
import pandas as pd

flat = []
for c in ordered_candidates:
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    flat.append({
        "candidate_id": c["candidate_id"],
        "current_title": p.get("current_title", ""),
        "years_of_experience": p.get("years_of_experience", 0),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "summary": p.get("summary", ""),
        "response_rate": s.get("recruiter_response_rate", 0.5),
        "last_active_date": s.get("last_active_date", ""),
        "notice_period_days": s.get("notice_period_days", 90),
        "open_to_work": s.get("open_to_work_flag", False),
        "avg_response_time_hours": s.get("avg_response_time_hours", 72),
        "interview_completion_rate": s.get("interview_completion_rate", 0.5),
        "github_activity_score": s.get("github_activity_score", -1),
        "verified_email": s.get("verified_email", False),
        "verified_phone": s.get("verified_phone", False),
        "linkedin_connected": s.get("linkedin_connected", False),
        "profile_completeness_score": s.get("profile_completeness_score", 50),
        "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d", 0),
        "preferred_work_mode": s.get("preferred_work_mode", "hybrid"),
        "willing_to_relocate": s.get("willing_to_relocate", False),
    })
pd.DataFrame(flat).to_parquet("artifacts/candidates_flat.parquet")
```

**Pickle (nested):**
```python
nested_data = {}
for c in ordered_candidates:
    cid = c["candidate_id"]
    career = c.get("career_history", [])
    nested_data[cid] = {
        "career_history": career,
        "career_text": " ".join((r.get("description", "") or "") for r in career),
        "career_companies": [r.get("company", "") for r in career],
        "education": c.get("education", []),
        "skill_names": [s.get("name", "") for s in c.get("skills", [])],
        "skill_assessment_scores": c.get("redrob_signals", {})
                                    .get("skill_assessment_scores", {}),
    }
pickle.dump(nested_data, open("artifacts/candidates_nested.pkl", "wb"))
```

### 2.11 Honeypot Pre-Detection

Run honeypot checks during precompute to avoid repeated `datetime.strptime()` parsing at ranking time (~3500 calls eliminated):

```python
from datetime import datetime

def precompute_honeypot_flags(ordered_candidates, skill_lookup):
    """
    Tiered honeypot detection. Returns dict with two sets:
    - 'hard': suspicion >= 3 → hard exclude (clearly fraudulent)
    - 'soft': suspicion == 2 → ×0.50 penalty (possibly messy data)
    False negatives (losing good candidates) hurt more than false positives.
    """
    hard_exclude = set()
    soft_penalize = set()
    for c in ordered_candidates:
        cid = c["candidate_id"]
        career = c.get("career_history", [])
        skills = skill_lookup.get(cid, [])
        suspicion = 0

        # Check 1: impossible skill claims (5+ expert, 3+ zero duration)
        experts = [s for s in skills if s["proficiency"] == "expert"]
        zero_dur = [s for s in experts if s["duration_months"] == 0]
        if len(experts) >= 5 and len(zero_dur) >= 3:
            suspicion += 1

        # Check 2: timeline mismatch (any role >12 month discrepancy)
        for role in career:
            sd = role.get("start_date")
            ed = role.get("end_date")
            dm = role.get("duration_months", 0)
            if sd and ed and dm:
                try:
                    s_dt = datetime.strptime(str(sd)[:10], "%Y-%m-%d")
                    e_dt = datetime.strptime(str(ed)[:10], "%Y-%m-%d")
                    actual = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                    if abs(actual - dm) > 12:
                        suspicion += 1
                        break
                except (ValueError, TypeError):
                    pass

        # Check 3: experience inflation
        p = c.get("profile", {})
        total_career = sum(r.get("duration_months", 0) for r in career)
        stated = p.get("years_of_experience", 0)
        if stated * 12 > total_career * 1.5 + 24:
            suspicion += 1

        if suspicion >= 3:
            hard_exclude.add(cid)
        elif suspicion == 2:
            soft_penalize.add(cid)

    logging.info(f"Honeypot: {len(hard_exclude)} hard-excluded, "
                 f"{len(soft_penalize)} soft-penalized")
    return {'hard': hard_exclude, 'soft': soft_penalize}

honeypot_data = precompute_honeypot_flags(ordered_candidates, skill_lookup)
pickle.dump(honeypot_data, open("artifacts/honeypot_flags.pkl", "wb"))
```

### 2.12 Artifact Budget

| Artifact | Size |
|---|---|
| models/bi/ + models/ce/ | ~160MB |
| candidate_embeddings.npy | ~77MB |
| candidate.index | ~150MB |
| candidate_order.npy | <1MB |
| archetype_max_scores.npy | <1MB |
| jd_*.npy, archetype_embeddings.npy | <100KB |
| skill_matches.pkl | ~50MB |
| candidates_flat.parquet | ~200MB |
| candidates_nested.pkl | ~400MB |
| honeypot_flags.pkl | <1KB |
| skill_threshold.npy | <1KB |
| **Total** | **~1.1GB** |

---

## 3. Ranking — Complete Implementation

### 3.1 Constants

```python
# --- KEYWORD LISTS ---

STRONG_KEYWORDS = [
    "ranking system", "ranking engine", "ranking model", "ranking pipeline",
    "search system", "search engine", "search platform", "search quality",
    "recommendation system", "recommendation engine", "recommender",
    "retrieval system", "retrieval pipeline", "information retrieval",
    "matching system", "matching engine", "candidate matching",
    "job matching", "talent matching", "relevance",
    "discovery platform", "personalization engine",
    "reranking", "re-ranking", "query understanding",
    "learning to rank",
]

MODERATE_KEYWORDS = [
    "embeddings", "vector search", "dense retrieval", "semantic search",
    "hybrid search", "neural search", "bm25", "inverted index",
    "ndcg", "mrr", "a/b test", "offline evaluation",
    "faiss", "pinecone", "milvus", "qdrant", "weaviate",
    "elasticsearch", "opensearch", "solr",
]

DEPLOYMENT_KEYWORDS = [
    "deployed to production", "shipped to production", "production system",
    "production environment", "real-time serving", "model serving",
    "live traffic", "production traffic", "launched", "went live",
    "serving users", "serving customers",
]

SCALE_KEYWORDS = [
    "at scale", "millions", "thousands of users", "daily active",
    "throughput", "latency", "sla", "high availability",
]

PRODUCT_ENG_KEYWORDS = [
    "api", "microservice", "endpoint", "ci/cd", "monitoring", "alerting",
]

ANTI_PRODUCTION_KEYWORDS = [
    "research only", "thesis", "proof of concept",
    "prototype only", "academic project",
]

VECTOR_TOOLS = [
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "chroma", "chromadb",
    "annoy", "scann", "vespa", "solr", "lucene", "pgvector",
]

VECTOR_SEARCH_JD_INDICES = {1, 15}

SERVICES_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "lti",
    "deloitte", "kpmg", "ey ", "ernst", "pwc",
]

TIER_A_TITLES = [
    "ml engineer", "machine learning", "ai engineer",
    "data scientist", "nlp engineer", "applied scientist",
    "research engineer", "deep learning", "search engineer",
    "recommendation", "retrieval",
]

TIER_B_TITLES = [
    "software engineer", "backend engineer", "full stack",
    "data engineer", "analytics engineer", "platform engineer",
]

TIER_C_TITLES = [
    "devops", "cloud engineer", "sre", "frontend",
    "qa engineer", "product manager",
]

# Hard-exclude these titles — they cannot be Senior ML Engineers
NON_TECH_TITLES = [
    "hr manager", "human resource", "recruiter",
    "marketing manager", "sales manager", "account manager",
    "operations manager", "accountant", "finance manager",
    "customer support", "copywriter", "graphic designer",
]

# But spare titles that contain tech qualifiers
TECH_CARVEOUTS = [
    "ml", "ai", "machine learning", "data", "engineer",
    "developer", "scientist", "nlp", "search", "recommendation",
    "retrieval", "talent intelligence", "talent tech",
]

PROF_WEIGHTS = {
    "expert": 1.0, "advanced": 0.85, "intermediate": 0.60, "beginner": 0.30
}

TITLE_LEVELS = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2, "scientist": 2,
    "senior": 3,
    "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "director": 5, "head": 5, "vp": 6,
    "founder": 5, "co-founder": 5, "cto": 6, "ceo": 6,
}

JD_CORE = """Senior AI Engineer, founding team. Must have shipped
ranking, search, or recommendation systems to production.
Production experience with embeddings, vector databases, hybrid search.
Python, PyTorch, evaluation frameworks. Product company background.
5-9 years experience. India preferred, hybrid work."""
```

### 3.2 Loading + Setup

```python
import numpy as np, pandas as pd, pickle, logging

logging.basicConfig(level=logging.INFO)

meta_df = pd.read_parquet("artifacts/candidates_flat.parquet")
nested = pickle.load(open("artifacts/candidates_nested.pkl", "rb"))
cand_embeds_f16 = np.load("artifacts/candidate_embeddings.npy")
archetype_max_raw = np.load("artifacts/archetype_max_scores.npy")
jd_queries = np.load("artifacts/jd_query_embeddings.npy")
archetypes = np.load("artifacts/archetype_embeddings.npy")
jd_skills = np.load("artifacts/jd_skill_embeddings.npy")
skill_lookup = pickle.load(open("artifacts/skill_matches.pkl", "rb"))
honeypot_data = pickle.load(open("artifacts/honeypot_flags.pkl", "rb"))
honeypot_hard = honeypot_data['hard']   # suspicion >= 3 → exclude
honeypot_soft = honeypot_data['soft']   # suspicion == 2 → ×0.50

# ORDERING ASSERTION
saved_order = np.load("artifacts/candidate_order.npy", allow_pickle=True)
assert len(cand_embeds_f16) == len(meta_df), "Embedding/metadata count mismatch"
assert np.array_equal(saved_order, meta_df["candidate_id"].values), \
    "Embedding/metadata ordering mismatch!"

# CAST FLOAT32 ONCE
cand_embeds = cand_embeds_f16.astype(np.float32)

# MAP ARCHETYPE MAX: [-1,1] → [0,1]
archetype_max = (archetype_max_raw.astype(np.float32) + 1.0) / 2.0

# VECTORIZED SEMANTIC: (100K, 5) — all cosine scores at once
all_semantic_raw = cand_embeds @ jd_queries.T
all_semantic = (all_semantic_raw + 1.0) / 2.0       # map to [0,1]
all_semantic_max = all_semantic.max(axis=1)           # best of 5 queries

# O(1) LOOKUPS
cid_to_idx = {cid: i for i, cid in enumerate(meta_df["candidate_id"])}
cid_to_years = dict(zip(meta_df["candidate_id"],
                         meta_df["years_of_experience"]))

# PRE-MATERIALIZE columns as numpy arrays — avoid repeated .iloc in loops
titles_arr = meta_df["current_title"].values
years_arr = meta_df["years_of_experience"].values
cids_arr = meta_df["candidate_id"].values

# PRE-COMPUTE timestamp — avoid repeated pd.Timestamp.now() in loops
TODAY = pd.Timestamp.now()

# SKILL THRESHOLD (calibrated during precompute)
try:
    SKILL_MATCH_THRESHOLD = float(np.load("artifacts/skill_threshold.npy")[0])
except FileNotFoundError:
    SKILL_MATCH_THRESHOLD = 0.50
    logging.warning("No calibrated threshold found, using default 0.50")

# FAISS
try:
    import faiss
    index = faiss.read_index("artifacts/candidate.index")
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    logging.warning("FAISS not available, using numpy fallback")
```

### 3.3 All Helper Functions

```python
# --- EXCLUSION CHECKS (return True = exclude this candidate) ---

def is_non_technical(title):
    """Hard exclude non-technical titles. Returns True if should exclude."""
    t = title.lower()
    is_non_tech = any(kw in t for kw in NON_TECH_TITLES)
    has_tech = any(kw in t for kw in TECH_CARVEOUTS)
    return is_non_tech and not has_tech

def is_honeypot(cid):
    """O(1) lookup — hard exclude only for suspicion >= 3."""
    return cid in honeypot_hard

def honeypot_penalty(cid):
    """Soft penalty for suspicion == 2. Returns multiplier."""
    if cid in honeypot_soft:
        return 0.50
    return 1.0

# --- SCORING HELPERS ---

def classify_title(title):
    """Returns 1.0 (Tier A), 0.6 (B), 0.3 (C), 0.05 (other)."""
    t = title.lower()
    if any(k in t for k in TIER_A_TITLES): return 1.0
    if any(k in t for k in TIER_B_TITLES): return 0.6
    if any(k in t for k in TIER_C_TITLES): return 0.3
    return 0.05

def score_experience(years):
    """Score experience band. JD sweet spot: 5-9 years."""
    if 5.0 <= years <= 9.0: return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 12.0: return 0.75
    if 3.0 <= years < 4.0 or 12.0 < years <= 15.0: return 0.5
    return 0.25

def compute_domain_keyword_score(career_text_lower):
    """Score domain-specific keyword matches. Returns [0, 1]."""
    score = 0.0
    for kw in STRONG_KEYWORDS:
        if kw in career_text_lower:
            score += 0.20
    for kw in MODERATE_KEYWORDS:
        if kw in career_text_lower:
            score += 0.10
    return min(score, 1.0)

def compute_recency_scores(last_active_series):
    """Vectorized recency scoring for entire DataFrame column."""
    dates = pd.to_datetime(last_active_series, errors='coerce')
    days = (TODAY - dates).dt.days.fillna(365)
    scores = pd.Series(0.1, index=days.index)
    scores[days <= 30] = 1.0
    scores[(days > 30) & (days <= 90)] = 0.8
    scores[(days > 90) & (days <= 180)] = 0.5
    scores[(days > 180) & (days <= 365)] = 0.25
    return scores

def compute_recency_score_single(last_active):
    """Single candidate recency. Uses pre-computed TODAY. Returns [0, 1]."""
    try:
        days = (TODAY - pd.to_datetime(last_active)).days
    except:
        return 0.5
    if days <= 30: return 1.0
    if days <= 90: return 0.8
    if days <= 180: return 0.5
    if days <= 365: return 0.25
    return 0.1

def compute_quick_behavioral(df):
    """Vectorized quick behavioral score for Stage 1 retrieval."""
    recency = compute_recency_scores(df["last_active_date"])
    response = df["response_rate"].clip(0, 1).fillna(0.5)
    otw = df["open_to_work"].astype(float).fillna(0.5)
    return 0.50 * response + 0.30 * recency + 0.20 * otw

def safe_val(val, default=0.5):
    """Safe value extraction — missing/NaN → default (0.5 = neutral)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

def endorse_weight(e):
    """Endorsement count → credibility weight."""
    if e == 0: return 0.40
    if e <= 5: return 0.70
    if e <= 20: return 0.90
    return 1.0

def duration_weight(d):
    """Skill duration (months) → credibility weight."""
    if d == 0: return 0.20
    if d <= 6: return 0.50
    if d <= 24: return 0.80
    return 1.0

def compute_credibility(skill, cid):
    """
    Weighted average credibility. NOT multiplicative.
    Minimum realistic output: ~0.33 (all minimums).
    Maximum: 1.0.
    """
    prof = PROF_WEIGHTS.get(skill["proficiency"], 0.60)
    endorse = endorse_weight(skill["endorsements"])
    dur = duration_weight(skill["duration_months"])

    assessments = nested.get(cid, {}).get("skill_assessment_scores", {})
    assess_score = assessments.get(skill["skill_name"])
    if assess_score is not None:
        assess = 1.0 if assess_score >= 70 else 0.80 if assess_score >= 40 else 0.50
    else:
        assess = 0.80

    credibility = 0.35 * prof + 0.25 * endorse + 0.25 * dur + 0.15 * assess
    return min(credibility, 1.0)

# --- PENALTIES (multiplicative, applied in final scoring) ---

def services_penalty(cid, f1_score):
    """
    Conditional services penalty. JD: "only worked at consulting firms — will not move forward."
    But a genuine ML engineer at HCL who built ranking systems shouldn't be crushed.
    - All services + low domain evidence (f1 < 0.30): ×0.40 (TCS COBOL developer)
    - All services + some domain evidence (f1 >= 0.30): ×0.80 (Mindtree ML engineer)
    """
    companies = nested.get(cid, {}).get("career_companies", [])
    non_empty = [co for co in companies if co.strip()]
    if not non_empty:
        return 1.0
    all_svc = all(
        any(sf in co.lower() for sf in SERVICES_FIRMS) for co in non_empty
    )
    if all_svc:
        if f1_score < 0.30:
            return 0.40  # no domain evidence + all services → heavy penalty
        return 0.80      # has domain evidence → moderate penalty
    return 1.0

def stuffer_check(cid, idx):
    """
    ×0.20 for non-tech-adjacent titles with many JD-matching skills
    but low average credibility. Catches keyword stuffers.
    Threshold: 0.45 (minimum credibility ~0.33, so this fires on weak claims).
    """
    title_tier = classify_title(str(titles_arr[idx]))
    if title_tier > 0.3:  # Tier A or B — unlikely stuffer
        return 1.0

    skills = skill_lookup.get(cid, [])
    matched = [s for s in skills if s["best_jd_match_score"] > SKILL_MATCH_THRESHOLD]
    if len(matched) < 5:
        return 1.0

    mean_cred = sum(compute_credibility(s, cid) for s in matched) / len(matched)
    if mean_cred < 0.45:
        return 0.20
    return 1.0
```

### 3.4 Stage 1: Union Retrieval + Scoring

```python
# --- Build retrieval pools ---

# Semantic pool (FAISS or numpy)
semantic_pool = set()
if USE_FAISS:
    for q in jd_queries:
        q32 = q.reshape(1, -1).astype(np.float32)
        _, indices = index.search(q32, 1000)
        semantic_pool.update(indices[0].tolist())
else:
    for i in range(5):
        sims = all_semantic_raw[:, i]  # raw cosine, fine for ranking
        top_k_idx = np.argpartition(sims, -1000)[-1000:]
        semantic_pool.update(top_k_idx.tolist())

# Skill-match pool
must_have_counts = np.zeros(len(meta_df))
for i, cid in enumerate(meta_df["candidate_id"]):
    skills = skill_lookup.get(cid, [])
    must_have_counts[i] = sum(1 for s in skills
        if s["best_jd_req_idx"] < 10 and s["best_jd_match_score"] > SKILL_MATCH_THRESHOLD)
skill_top500 = set(np.argpartition(must_have_counts, -500)[-500:].tolist())

# Behavioral pool
behavioral_quick = compute_quick_behavioral(meta_df)
behavioral_arr = behavioral_quick.values  # pre-materialize for loops
behav_top200 = set(np.argpartition(behavioral_arr, -200)[-200:].tolist())

# UNION
candidate_pool = semantic_pool | skill_top500 | behav_top200
logging.info(f"Union pool size: {len(candidate_pool)}")

# --- HARD EXCLUSIONS before scoring ---
excluded_honeypots = 0
excluded_titles = 0
filtered_pool = []

for idx in candidate_pool:
    cid = cids_arr[idx]       # pre-materialized numpy array
    title = str(titles_arr[idx])  # pre-materialized numpy array

    if is_honeypot(cid):
        excluded_honeypots += 1
        continue

    if is_non_technical(title):
        excluded_titles += 1
        continue

    filtered_pool.append(idx)

logging.info(f"Excluded {excluded_honeypots} honeypots, "
             f"{excluded_titles} non-technical titles. "
             f"Remaining: {len(filtered_pool)}")

# --- Stage 1 scoring (uses pre-materialized arrays, no .iloc) ---
def stage1_score(idx):
    cid = cids_arr[idx]

    semantic = float(all_semantic_max[idx])        # pre-computed [0,1]
    archetype = float(archetype_max[idx])           # pre-computed [0,1]

    skills = skill_lookup.get(cid, [])
    matched = [s for s in skills
               if s["best_jd_req_idx"] < 10 and s["best_jd_match_score"] > 0.50]
    skill = min(len(matched) / 10.0, 1.0)

    career_text = nested.get(cid, {}).get("career_text", "").lower()
    pattern = compute_domain_keyword_score(career_text)

    title = classify_title(str(titles_arr[idx]))
    behav = float(behavioral_arr[idx])  # pre-materialized
    exp = score_experience(float(years_arr[idx]))

    score = (
        0.25 * semantic +
        0.15 * archetype +
        0.20 * skill +
        0.15 * pattern +
        0.10 * title +
        0.08 * behav +
        0.07 * exp
    )
    score *= stuffer_check(cid, idx)
    # services_penalty NOT applied at Stage 1
    # Reason: real f1 isn't available yet (cross-encoder hasn't run).
    # Using keyword proxy could misclassify genuine ML engineers at services
    # firms who use non-standard language. Applied only in final scoring.
    return score

pool_scores = [(idx, cids_arr[idx], stage1_score(idx))
               for idx in filtered_pool]
pool_scores.sort(key=lambda x: x[2], reverse=True)

top_k = min(800, len(pool_scores))
top_800 = pool_scores[:top_k]
logging.info(f"Stage 1 selected top {top_k}")
```

### 3.5 Stage 2: Cross-Encoder (Sigmoid)

```python
def select_relevant_roles(career_history, n=3):
    """Pick n roles most relevant to retrieval/ranking domain."""
    if len(career_history) <= n:
        return career_history

    scored = []
    for role in career_history:
        desc = (role.get("description", "") or "").lower()
        relevance = sum(1 for kw in STRONG_KEYWORDS if kw in desc)
        relevance += sum(0.5 for kw in MODERATE_KEYWORDS if kw in desc)
        scored.append((relevance, role))
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored[0][0] == 0:
        by_date = sorted(career_history,
                         key=lambda r: r.get("start_date", ""),
                         reverse=True)
        return by_date[:n]
    return [role for _, role in scored[:n]]

def build_ce_text(cid):
    """Build candidate text for cross-encoder input (~375 tokens)."""
    data = nested.get(cid, {})
    parts = []

    summary = str(meta_df.iloc[cid_to_idx[cid]].get("summary", "") or "")
    if summary:
        parts.append(summary[:200])

    career = data.get("career_history", [])
    relevant = select_relevant_roles(career, n=3)
    for role in relevant:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = (role.get("description", "") or "")[:250]
        parts.append(f"{title} at {company}: {desc}")

    skill_names = ", ".join(data.get("skill_names", [])[:8])
    parts.append(f"Skills: {skill_names}")

    return " ".join(parts)[:1500]

# --- Run cross-encoder ---
from sentence_transformers import CrossEncoder
ce_model = CrossEncoder("artifacts/models/ce/", device="cpu")

pairs = [(JD_CORE, build_ce_text(cid)) for _, cid, _ in top_800]
raw_logits = ce_model.predict(pairs, batch_size=32)

# SIGMOID normalization — preserves magnitude, no artificial separation
cross_scores = 1.0 / (1.0 + np.exp(-raw_logits))
```

### 3.6 Stage 3: 7-Feature Scoring

**Every feature clamped to [0.0, 1.0]. All cosine mapped via (cos+1)/2.**

```python
# --- Feature 1: Career Domain Evidence (weight: 0.28) ---

def feature_1(idx, cid, cross_sigmoid):
    # Sub-A: Cross-encoder sigmoid (already [0,1])

    # Sub-B: Archetype max (precomputed, already [0,1])
    arch = float(archetype_max[idx])

    # Sub-C: 0.25 keyword + 0.75 semantic (domain-specific query only)
    career_text = nested.get(cid, {}).get("career_text", "").lower()
    kw = compute_domain_keyword_score(career_text)
    # Use ONLY query 0 ("search ranking recommendation retrieval...") for F1
    # all_semantic_max includes query 2 (production/DevOps) which inflates
    # non-domain candidates. Query 0 is domain-specific and more precise.
    cos_raw = float(cand_embeds[idx] @ jd_queries[0])
    sem = (cos_raw + 1.0) / 2.0

    domain = 0.25 * kw + 0.75 * sem

    result = 0.40 * cross_sigmoid + 0.30 * arch + 0.30 * domain
    return max(min(result, 1.0), 0.0)


# --- Feature 2: Retrieval/Search Expertise (weight: 0.30) ---
# Uses TOP-5 must-have + TOP-3 nice-to-have for DEPTH over breadth

def feature_2(cid):
    skills = skill_lookup.get(cid, [])

    # DEDUP by JD requirement index — keep highest credible score
    seen_jd_reqs = {}
    for s in skills:
        if s["best_jd_match_score"] < SKILL_MATCH_THRESHOLD:
            continue
        req_idx = s["best_jd_req_idx"]
        cred = compute_credibility(s, cid)
        credible_score = s["best_jd_match_score"] * cred
        if req_idx not in seen_jd_reqs or credible_score > seen_jd_reqs[req_idx]:
            seen_jd_reqs[req_idx] = credible_score

    must_hits = sorted([v for k, v in seen_jd_reqs.items() if k < 10],
                       reverse=True)
    nice_hits = sorted([v for k, v in seen_jd_reqs.items() if k >= 10],
                       reverse=True)

    # DEPTH: top-5 must-have scores, divisor = max(count, 3)
    top_must = must_hits[:5]
    depth = min(sum(top_must) / max(len(top_must), 3), 1.0) if top_must else 0.0

    # COVERAGE: unique JD must-have requirements matched
    # 6 out of 10 unique must-haves = perfect coverage
    # Rewards breadth alongside depth
    coverage = min(len(must_hits) / 6, 1.0)

    # NICE-TO-HAVE: top-3, divisor = max(count, 2)
    top_nice = nice_hits[:3]
    nice_score = min(sum(top_nice) / max(len(top_nice), 2), 1.0) if top_nice else 0.0

    # Blend: depth matters most, but coverage prevents FAISS×4 beating
    # FAISS+Elasticsearch+PyTorch+Embeddings
    must_component = 0.60 * depth + 0.40 * coverage
    result = 0.70 * must_component + 0.30 * nice_score
    return min(result, 1.0)


# --- Feature 3: Production Deployment (weight: 0.15) ---

def feature_3(cid, idx):
    ct = nested.get(cid, {}).get("career_text", "").lower()

    kw_score = 0.0
    for k in DEPLOYMENT_KEYWORDS:
        if k in ct: kw_score += 0.15
    for k in SCALE_KEYWORDS:
        if k in ct: kw_score += 0.10
    for k in PRODUCT_ENG_KEYWORDS:
        if k in ct: kw_score += 0.08
    for k in ANTI_PRODUCTION_KEYWORDS:
        if k in ct: kw_score -= 0.15
    kw_score = max(min(kw_score, 1.0), 0.0)

    cos_raw = float(cand_embeds[idx] @ jd_queries[2])
    sem = (cos_raw + 1.0) / 2.0

    result = 0.40 * kw_score + 0.60 * sem
    return max(min(result, 1.0), 0.0)


# --- Feature 4: Vector DB & Infrastructure (weight: 0.10) ---

def feature_4(cid, idx):
    skills = skill_lookup.get(cid, [])
    career_text = nested.get(cid, {}).get("career_text", "").lower()

    # TOOL-BASED COMPONENT (explicit tool names)
    tools_found = {}
    for s in skills:
        name_lower = s.get("norm_name", s["skill_name"].lower())

        # ONLY vector/search tools or specific JD indices
        is_tool = (name_lower in VECTOR_TOOLS or
                   s["best_jd_req_idx"] in VECTOR_SEARCH_JD_INDICES)
        if not is_tool:
            continue

        cred = compute_credibility(s, cid)
        in_desc = name_lower in career_text
        weight = 1.0 if in_desc else 0.6
        effective = cred * weight
        if name_lower not in tools_found or effective > tools_found[name_lower]:
            tools_found[name_lower] = effective

    n = len(tools_found)
    avg_cred = sum(tools_found.values()) / max(n, 1)
    base = {0: 0.0, 1: 0.4, 2: 0.7}.get(n, 1.0)
    tool_score = min(base * avg_cred, 1.0)

    # SEMANTIC COMPONENT — catches "built dense retrieval pipelines"
    # without explicitly mentioning FAISS/Pinecone/etc.
    # Uses JD query 1: "embeddings vector database FAISS... hybrid search"
    cos_raw = float(cand_embeds[idx] @ jd_queries[1])
    sem_infra = (cos_raw + 1.0) / 2.0

    # Blend: tools are stronger signal, semantic catches implicit experience
    result = 0.60 * tool_score + 0.40 * sem_infra
    return max(min(result, 1.0), 0.0)


# --- Feature 5: Availability & Behavioral (weight: 0.10, capped 0.80) ---

def feature_5(idx):
    row = meta_df.iloc[idx]

    recency = compute_recency_score_single(row["last_active_date"])

    rr = safe_val(row["response_rate"], 0.5)
    response = (1.0 if rr >= 0.7 else 0.85 if rr >= 0.5
                else 0.60 if rr >= 0.3 else 0.35 if rr >= 0.15 else 0.15)

    rt = safe_val(row["avg_response_time_hours"], 72)
    resp_time = 1.0 if rt < 24 else 0.8 if rt < 72 else 0.5 if rt < 168 else 0.3

    ic = safe_val(row["interview_completion_rate"], 0.5)
    interview = (1.0 if ic >= 0.8 else 0.75 if ic >= 0.6
                 else 0.5 if ic >= 0.4 else 0.25)

    otw = 1.0 if safe_val(row["open_to_work"], False) else 0.5

    gh = safe_val(row["github_activity_score"], -1)
    github = (0.3 if gh < 0 else 0.4 if gh <= 20
              else 0.7 if gh <= 50 else 0.9 if gh <= 80 else 1.0)

    ve = 1.0 if safe_val(row["verified_email"], False) else 0.0
    vp = 1.0 if safe_val(row["verified_phone"], False) else 0.0
    li = 1.0 if safe_val(row["linkedin_connected"], False) else 0.0
    trust = 0.3 * ve + 0.3 * vp + 0.4 * li

    nd = safe_val(row["notice_period_days"], 90)
    notice = (1.0 if nd <= 30 else 0.8 if nd <= 60
              else 0.6 if nd <= 90 else 0.4 if nd <= 120 else 0.25)

    pc = safe_val(row["profile_completeness_score"], 50) / 100
    sr = min(safe_val(row["saved_by_recruiters_30d"], 0) / 15, 1.0)

    result = (
        0.22 * recency + 0.22 * response + 0.10 * resp_time +
        0.10 * interview + 0.08 * otw + 0.08 * github +
        0.05 * trust + 0.08 * notice + 0.04 * pc + 0.03 * sr
    )
    return min(result, 0.80)  # CAPPED


# --- Feature 6: LLM & Adjacent (weight: 0.04) ---

def feature_6(cid):
    skills = skill_lookup.get(cid, [])
    VALUED = {10, 11, 12}
    LESS_VALUED = {13, 14}

    score = 0.0
    for s in skills:
        if s["best_jd_match_score"] < SKILL_MATCH_THRESHOLD:
            continue
        cred = compute_credibility(s, cid)
        if s["best_jd_req_idx"] in VALUED:
            score += 0.15 * cred
        elif s["best_jd_req_idx"] in LESS_VALUED:
            score += 0.05 * cred
    return min(score, 1.0)


# --- Feature 7: Career Progression (weight: 0.03) ---

def get_title_level(title):
    t = title.lower()
    for kw in sorted(TITLE_LEVELS.keys(), key=len, reverse=True):
        if kw in t:
            return TITLE_LEVELS[kw]
    return 2

def feature_7(cid):
    career = nested.get(cid, {}).get("career_history", [])
    if len(career) < 2:
        return 0.5  # neutral

    # SORT CHRONOLOGICALLY — oldest first
    career_sorted = sorted(career, key=lambda r: r.get("start_date", ""))
    levels = [get_title_level(r.get("title", "")) for r in career_sorted]

    ups = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i-1])
    downs = sum(1 for i in range(1, len(levels)) if levels[i] < levels[i-1])
    total = len(levels) - 1

    if ups > downs:
        return min(0.5 + (ups / total) * 0.5, 1.0)
    elif ups == downs:
        return 0.5
    else:
        return max(0.5 - (downs / total) * 0.3, 0.2)
```

### 3.7 Final Scoring Formula

```python
def compute_final_score(idx, cid, cross_sigmoid):
    f1 = feature_1(idx, cid, cross_sigmoid)
    f2 = feature_2(cid)
    f3 = feature_3(cid, idx)
    f4 = feature_4(cid, idx)
    f5 = feature_5(idx)
    f6 = feature_6(cid)
    f7 = feature_7(cid)

    raw = (
        0.32 * f1 +    # career domain evidence (highest — resists skill stuffing)
        0.26 * f2 +    # retrieval expertise (depth + coverage)
        0.15 * f3 +    # production deployment (keyword + semantic)
        0.10 * f4 +    # vector DB (tool + semantic)
        0.10 * f5 +    # availability (capped 0.80)
        0.04 * f6 +    # LLM adjacent
        0.03 * f7      # career progression
    )
    # Weights sum to 1.00
    # NO additive title/experience modifiers — already in Stage 1

    # MULTIPLICATIVE PENALTIES ONLY
    # Honeypot hard-excluded before this point; soft-penalized here
    raw *= services_penalty(cid, f1)
    raw *= stuffer_check(cid, idx)
    raw *= honeypot_penalty(cid)  # ×0.50 for suspicion==2

    return raw, [f1, f2, f3, f4, f5, f6, f7]

# --- Score top 800 → select top 100 ---
final_scores = []
for rank_in_800, (idx, cid, s1) in enumerate(top_800):
    score, features = compute_final_score(idx, cid, cross_scores[rank_in_800])
    final_scores.append((idx, cid, score, features))

final_scores.sort(key=lambda x: x[2], reverse=True)
top_100 = final_scores[:100]
```

---

## 4. Stage 4: Data-Driven Reasoning

```python
def generate_reasoning(cid, rank, features, idx):
    data = nested.get(cid, {})
    row = meta_df.iloc[idx]

    title = str(row["current_title"])
    years = round(float(row["years_of_experience"]), 1)
    country = str(row.get("country", ""))
    location = f"{row['location']}, {country}"
    notice = int(safe_val(row.get("notice_period_days", 90), 90))
    response_rate = float(safe_val(row.get("response_rate", 0.5), 0.5))
    companies = data.get("career_companies", [])[:3]
    top_skills = data.get("skill_names", [])[:5]
    career_text = data.get("career_text", "").lower()

    # --- SENTENCE 1: Primary strength (highest feature) ---
    sorted_f = sorted(enumerate(features), key=lambda x: x[1], reverse=True)
    best_i, best_v = sorted_f[0]
    s1 = _strength_sentence(best_i, cid, data, title, years,
                            companies, top_skills, career_text)

    # --- SENTENCE 2: Secondary strength (if significant) ---
    second_i, second_v = sorted_f[1]
    s2 = ""
    if second_v > 0.3:
        s2 = _secondary_sentence(second_i, cid, data, top_skills, career_text)

    # --- SENTENCE 3: Concern (data-driven) ---
    s3 = _concern_sentence(notice, years, response_rate,
                           country, location, companies)

    # --- Assemble by rank band (controls length/tone, NOT structure) ---
    if rank <= 10:
        parts = [s for s in [s1, s2, s3] if s]
    elif rank <= 30:
        parts = [s1, s3] if s3 else [s1, s2] if s2 else [s1]
    elif rank <= 60:
        parts = [s1, s3] if s3 else [s1]
    else:
        parts = [s1]
        if s3: parts.append(s3)

    return " ".join(parts)


def _strength_sentence(feat_idx, cid, data, title, years,
                       companies, top_skills, career_text):
    """Primary strength — built from actual profile data."""
    co = companies[0] if companies else "their company"

    if feat_idx == 0:  # career domain
        found = [kw for kw in STRONG_KEYWORDS if kw in career_text]
        if found:
            return (f"{title} with {years}yrs; career includes "
                    f"{found[0]} work at {co}.")
        return f"{title} with {years}yrs; strong alignment with retrieval/ranking requirements."

    elif feat_idx == 1:  # retrieval skills
        matched = [s["skill_name"] for s in skill_lookup.get(cid, [])
                   if s["best_jd_match_score"] > 0.50
                   and s["best_jd_req_idx"] < 10][:4]
        skills_str = ", ".join(matched) if matched else ", ".join(top_skills[:3])
        return f"Credibility-verified retrieval skills: {skills_str}; {years}yrs as {title}."

    elif feat_idx == 2:  # production
        found = [kw for kw in DEPLOYMENT_KEYWORDS if kw in career_text]
        if found:
            return f"Production deployment evidence ({found[0]}) at {co}; {years}yrs experience."
        return f"Career at {co} indicates production-level work; {years}yrs as {title}."

    elif feat_idx == 3:  # vector DB
        tools = [s["skill_name"] for s in skill_lookup.get(cid, [])
                 if s.get("norm_name", s["skill_name"].lower()) in VECTOR_TOOLS][:3]
        tools_str = ", ".join(tools) if tools else "vector search tools"
        return f"Search infrastructure experience with {tools_str}; {years}yrs as {title}."

    elif feat_idx == 4:  # behavioral
        return f"Highly active on Redrob platform; {title} with {years}yrs at {co}."

    elif feat_idx == 5:  # LLM
        return f"LLM/adjacent ML expertise complements core skills; {title} with {years}yrs."

    else:  # progression
        return f"Strong career progression trajectory; currently {title} with {years}yrs."


def _secondary_sentence(feat_idx, cid, data, top_skills, career_text):
    """Supporting evidence — shorter, complementary to primary."""
    if feat_idx == 0:
        found = [kw for kw in STRONG_KEYWORDS if kw in career_text]
        return f"Also shows {found[0]} domain experience." if found else ""

    elif feat_idx == 1:
        matched = [s["skill_name"] for s in skill_lookup.get(cid, [])
                   if s["best_jd_match_score"] > 0.50][:3]
        return f"Relevant skills include {', '.join(matched)}." if matched else ""

    elif feat_idx == 2:
        return "Has shipped systems to production environments."

    elif feat_idx == 3:
        tools = [s["skill_name"] for s in skill_lookup.get(cid, [])
                 if s.get("norm_name", s["skill_name"].lower()) in VECTOR_TOOLS][:2]
        return f"Experience with {', '.join(tools)}." if tools else ""

    elif feat_idx == 4:
        return "Active platform engagement signals availability."

    elif feat_idx == 5:
        return "Additional LLM fine-tuning experience."

    else:
        return "Consistent career growth trajectory."


def _concern_sentence(notice, years, response_rate, country, location, companies):
    """Build concern from actual data — returns the most relevant one."""
    if notice > 90:
        return f"Note: {notice}-day notice period significantly exceeds JD's 30-day preference."
    if notice > 60:
        return f"Note: {notice}-day notice period exceeds JD's 30-day preference."
    if years < 4:
        return f"Note: {years}yrs experience is below the 5-9yr target range."
    if years > 12:
        return f"Note: {years}yrs experience is above the 5-9yr target range."
    if response_rate < 0.2:
        return f"Note: low recruiter response rate ({response_rate:.0%}) may indicate limited availability."
    if "india" not in country.lower():
        return f"Note: based in {location}, outside preferred India locations."

    non_empty = [co for co in companies if co.strip()]
    if non_empty:
        all_svc = all(any(sf in co.lower() for sf in SERVICES_FIRMS) for co in non_empty)
        if all_svc:
            return f"Note: entire career at services firms ({', '.join(str(c) for c in non_empty[:2])})."

    return ""
```

---

## 5. Output

```python
import csv

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # CRITICAL: column order must be candidate_id, rank, score, reasoning
    # Validator does strict header equality — wrong order = auto-rejection
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (idx, cid, score, features) in enumerate(top_100, 1):
        reasoning = generate_reasoning(cid, rank, features, idx)
        writer.writerow([cid, rank, round(score, 6), reasoning])

logging.info("submission.csv written with 100 candidates")
```

---

## 6. Missing-ID Handling

```python
# At any point where we access precomputed data:
# If candidate_id is not in artifacts, skip + log.

def safe_lookup(cid):
    if cid not in cid_to_idx:
        logging.warning(f"Candidate {cid} not in artifacts. Skipping.")
        return None
    return cid_to_idx[cid]
```

---

## 7. File Structure

```
redrob challange/
├── precompute.py
├── rank.py
├── scoring/
│   ├── __init__.py
│   ├── constants.py        All keyword lists, JD queries, thresholds
│   ├── features.py         7 features (all clamped, cosine mapped)
│   ├── penalties.py        services ×0.40, stuffer ×0.20
│   ├── exclusions.py       is_honeypot(), is_non_technical()
│   └── reasoning.py        Data-driven reasoning builder
├── artifacts/              ~1.1GB
├── requirements.txt
└── README.md
```

## 8. Dependencies

```
sentence-transformers>=2.2.0
numpy>=1.24.0
torch>=2.0.0
pandas>=2.0.0
pyarrow>=12.0.0
faiss-cpu>=1.7.0
```

## 9. Runtime Budget

| Step | Time |
|---|---|
| Load + cast + vectorize | ~22s |
| FAISS retrieval | ~1s |
| Skill/behavioral retrieval | ~5s |
| Hard exclusions (~3500) | ~1s |
| Stage 1 scoring (remaining) | ~12s |
| Load cross-encoder + cold torch | ~20s |
| Cross-encoder **800** pairs | ~95s |
| Feature scoring (800) | ~15s |
| Reasoning (100) | ~5s |
| Write CSV | <1s |
| **Total** | **~177s** |
| **Worst case** | **~260s** |
| **Limit** | **300s ✅** |

## 10. Changes v14 → v15

| Issue | v14 | v15 |
|---|---|---|
| Honeypot exclusion | suspicion ≥2 → hard exclude all | **≥3 hard exclude, ==2 → ×0.50 penalty** |
| Feature weights | F1=0.28, F2=0.30 | **F1=0.32, F2=0.26** (career > skills) |
| Stage 2 pool | Top-500 | **Top-800** (better recall) |
| Cross-encoder pairs | 500 | **800** (~95s, still under budget) |

## 11. Cumulative Changes (v8 → v15)

| Issue | v8 | v15 |
|---|---|---|
| CSV column order | Wrong (`rank` first) | **`candidate_id` first** |
| Cross-encoder normalization | Percentile | **Sigmoid** |
| Cosine range | [-1,1] assumed [0,1] | **(cos+1)/2 everywhere** |
| Title/exp double-counting | Stage 1 + final | **Stage 1 only** |
| Non-tech titles | No penalty | **Hard exclude** |
| Stuffer threshold | 0.15 (never fires) | **0.45** |
| Feature weights | F1=0.35, F2=0.25 | **F1=0.32, F2=0.26** (career dominates) |
| Feature 1 semantic | Single query | **Query 0 only** (domain-specific) |
| Feature 2 scoring | sum(all)/10 | **0.60 depth + 0.40 coverage** |
| Feature 4 | `match_score > 0.60` | **VECTOR_TOOLS or JD {1,15} + 40% semantic** |
| Skill threshold | Hardcoded 0.50 | **Calibrated, used everywhere** |
| Honeypot | ×0.30 soft | **≥3 hard exclude, ==2 ×0.50, precomputed** |
| Services | ×0.92 | **×0.40 if weak, ×0.80 if strong** (final only) |
| Reasoning | Template rotation | **Data-driven** |
| Stage 2 pool | 500 | **800** (better recall) |
| Candidate embedding | Unstructured concat | **Labeled sections** |
| Semantic scoring | Per-candidate loop | **Vectorized matmul** |
| Array access | `.iloc` in loops | **Pre-materialized numpy** |
| Timestamp | `now()` in loops | **Pre-computed `TODAY`** |

## 12. Verification Checklist

```bash
python precompute.py --candidates ./candidates.jsonl --out-dir ./artifacts/
time python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- [ ] **CSV header is `candidate_id,rank,score,reasoning`** (auto-reject check)
- [ ] Ordering assertion passes at load
- [ ] Runtime < 300s on cold CPU
- [ ] No network calls (airplane mode)
- [ ] Zero hard-honeypots (suspicion≥3) in top 100
- [ ] Soft-honeypots (suspicion==2) penalized ×0.50, not excluded
- [ ] Zero HR/Marketing/Accountant in top 100
- [ ] Top 10 are ML/AI/Search engineers with retrieval evidence
- [ ] F1 weight (0.32) > F2 weight (0.26) verified
- [ ] Feature 2 test: 3 strong skills > 5 weak skills
- [ ] Feature 4 test: FAISS+Elasticsearch HIGH, PyTorch-only LOW
- [ ] Stuffer check fires on test case (Tier-C + AI keywords + low cred)
- [ ] Services: all-TCS w/ low f1 → ×0.40, all-HCL w/ high f1 → ×0.80
- [ ] 10 reasoning samples: structurally different, facts verifiable
- [ ] Scores monotonically non-increasing
- [ ] Cold Docker test: `docker run --network none <img> time python rank.py`


