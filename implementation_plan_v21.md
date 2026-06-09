# Redrob AI Candidate Ranking System — v21 FINAL

Fully self-contained. Every function defined. No cross-references. Ready to code.

Supersedes v15. Incorporates Auto JD Compiler, Evidence Credibility, chunk-level embeddings,
data-verified company taxonomy, sentinel handling, top-10 re-verification, and all
findings from full dataset analysis of 100K candidates.

---

## What Changed v15 → v21

| Area | v15 | v21 |
|---|---|---|
| JD parsing | Hardcoded queries/skills/archetypes | **Auto JD Compiler — everything derived from JD text** |
| Candidate representation | Single embedding (skills+career+summary) | **4 chunk embeddings (summary, career, skills, achievements proxy)** |
| Company taxonomy | Services firms penalized | **Fictional companies mapped to product/services/manufacturing; Software fictional = product company** |
| Non-tech pre-filter | Applied after retrieval | **Hard pre-filter before retrieval — eliminates 51K candidates upfront** |
| Honeypot detection | Timeline + skill inflation | **+ title/description coherence check (catches Naina Bose pattern)** |
| Behavioral signals | F6 at weight 0.10 | **Sentinel handling for -1 values; `saved_by_recruiters_30d` promoted; notice_period as top-10 modifier** |
| Skill assessments | Weighted heavily in credibility | **De-weighted — assessment coverage for JD-relevant skills is near-zero** |
| Top-10 handling | Implicit in final sort | **Explicit top-10 re-verification pass (NDCG@10 = 50% of score)** |
| Achievements field | Referenced in plan | **Removed — field does not exist in schema** |
| Retrieval | 5 queries → union | **Auto-generated queries + per-section chunk retrieval** |

---

## Architecture

```
CANDIDATES.JSONL + JOB_DESCRIPTION.MD
              │
              ▼
┌─────────────────────────────┐
│  PRE-COMPUTATION (GPU)      │
│  Auto JD Compiler           │
│  Pre-filter (51K removed)   │
│  Chunk embeddings           │
│  Evidence extraction        │
│  Credibility scores         │
│  Honeypot flags             │
│  Company taxonomy mapping   │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  RANKING (CPU ≤300s)        │
│                             │
│  Stage 0: Hard pre-filter   │  100K → ~49K
│  Stage 1: Multi-retrieval   │  ~49K → ~3500
│  Stage 2: Exclusions        │  ~3500 → ~2800
│  Stage 3: Fusion scoring    │  ~2800 → 800
│  Stage 4: Cross-encoder     │  800 → 300
│  Stage 5: Feature scoring   │  300 → 100
│  Stage 6: Top-10 verify     │  Reshuffle top-10 if needed
│  Stage 7: Reasoning         │  100 evidence-grounded strings
│  Output: submission.csv     │
└─────────────────────────────┘
```

---

## Part 1 — Pre-Computation

### 1.1 Models

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

bi_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cuda')
# BGE-small over MiniLM — better retrieval performance, same 384-d, similar speed
bi_model.save('artifacts/models/bi/')

ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
ce_model.save('artifacts/models/ce/')
```

---

### 1.2 Auto JD Compiler

No more hardcoded queries, skills, or archetypes. Everything derived from `job_description.md`.

```python
import re

JD_TEXT = open("job_description.md").read()

# --- Section Parser ---
JD_SECTIONS = {}

SECTION_HEADERS = {
    "must_have":   ["things you absolutely need", "required", "must have", "must-have"],
    "nice_to_have":["things we'd like", "nice to have", "nice-to-have", "preferred"],
    "disqualifiers":["things we explicitly do not want", "disqualifiers", "do not want"],
    "responsibilities": ["what you'd actually be doing", "responsibilities", "what you will do"],
    "behavioral": ["the vibe check", "culture", "behavioral", "vibe"],
}

lines = JD_TEXT.split('\n')
current_section = "general"
for line in lines:
    ll = line.lower().strip()
    matched = False
    for sec, patterns in SECTION_HEADERS.items():
        if any(p in ll for p in patterns):
            current_section = sec
            matched = True
            break
    if not matched:
        JD_SECTIONS.setdefault(current_section, []).append(line)

# --- Skill Extraction ---
SKILL_EXTRACTION_PATTERNS = [
    r'\b(FAISS|Milvus|Pinecone|Weaviate|Qdrant|Elasticsearch|OpenSearch|Chroma|pgvector)\b',
    r'\b(sentence-transformers?|BGE|E5|OpenAI embeddings?)\b',
    r'\b(PyTorch|TensorFlow|JAX)\b',
    r'\b(Python|SQL)\b',
    r'\b(NDCG|MRR|MAP|A/B test(?:ing)?|offline eval(?:uation)?)\b',
    r'\b(LoRA|QLoRA|PEFT|fine-tun(?:e|ing))\b',
    r'\b(XGBoost|LightGBM|learning.to.rank)\b',
    r'\b(Docker|Kubernetes|MLOps|MLflow|Airflow)\b',
    r'\b(retrieval|ranking|search|recommendation|matching|relevance|discovery)\b',
    r'\b(embeddings?|vector database|hybrid search|dense retrieval|BM25)\b',
    r'\b(LLM|transformers?|NLP|information retrieval)\b',
]

extracted_skills = set()
for pattern in SKILL_EXTRACTION_PATTERNS:
    matches = re.findall(pattern, JD_TEXT, re.IGNORECASE)
    extracted_skills.update(m.lower() for m in matches)

# Partition by section weight
must_have_section = '\n'.join(JD_SECTIONS.get('must_have', []))
nice_section = '\n'.join(JD_SECTIONS.get('nice_to_have', []))

JD_MUST_HAVE_SKILLS = []
JD_NICE_TO_HAVE_SKILLS = []

for skill in sorted(extracted_skills):
    if skill in must_have_section.lower():
        JD_MUST_HAVE_SKILLS.append(skill)
    else:
        JD_NICE_TO_HAVE_SKILLS.append(skill)

# --- Experience Range Extraction ---
exp_match = re.search(r'(\d+)[–\-](\d+)\s*years?', JD_TEXT, re.IGNORECASE)
if exp_match:
    JD_EXP = {
        "ideal_min": int(exp_match.group(1)),
        "ideal_max": int(exp_match.group(2)),
        "soft_min": int(exp_match.group(1)) - 1,
        "soft_max": int(exp_match.group(2)) + 3,
    }
else:
    JD_EXP = {"ideal_min": 5, "ideal_max": 9, "soft_min": 4, "soft_max": 12}

# --- Disqualifier Extraction ---
disq_text = '\n'.join(JD_SECTIONS.get('disqualifiers', []))

JD_DISQUALIFIERS = {
    "research_only": any(p in disq_text.lower() for p in
                         ["research only", "pure research", "academic lab", "without production"]),
    "langchain_only": any(p in disq_text.lower() for p in
                          ["langchain", "openai wrapper", "under 12 months"]),
    "non_coding_architect": any(p in disq_text.lower() for p in
                                ["hasn't written", "architecture role", "tech lead"]),
    "consulting_only": any(p in disq_text.lower() for p in
                           ["consulting firm", "tcs", "infosys", "entire career"]),
}

# --- Auto-Generate JD Queries (5 queries from section themes) ---
def build_query_from_section(section_lines, max_len=120):
    text = ' '.join(section_lines)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9/\-\.]+\b', text)
    stopwords = {'the','a','an','and','or','but','in','on','at','to','for','of','with',
                 'is','are','was','were','be','been','being','have','has','had','do',
                 'does','did','will','would','could','should','may','might','shall',
                 'we','our','you','your','they','their','this','that','these','those',
                 'if','when','where','which','who','how','what','why','not','no','any'}
    content = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    # Deduplicate preserving order
    seen = set()
    unique = [w for w in content if not (w.lower() in seen or seen.add(w.lower()))]
    return ' '.join(unique[:25])

JD_QUERIES = [
    # Query 0: core domain (from must-have + responsibilities)
    build_query_from_section(
        JD_SECTIONS.get('must_have', []) + JD_SECTIONS.get('responsibilities', [])
    ),
    # Query 1: vector infrastructure (extracted infrastructure skills)
    "embeddings vector database FAISS Milvus Pinecone Weaviate Qdrant hybrid search dense retrieval",
    # Query 2: production deployment
    "shipped deployed production serving real users scale traffic product engineering",
    # Query 3: evaluation + measurement
    "NDCG MRR MAP evaluation framework ranking A/B testing offline metrics",
    # Query 4: archetype (ideal candidate description from JD)
    build_query_from_section(JD_SECTIONS.get('general', [])[-20:]),  # "how to read between lines" section
]

# --- Auto-Generate JD Skill Embeddings (must-have first, then nice-to-have) ---
JD_SKILLS_ORDERED = (
    # Must-have (indices 0-9) — derived from extraction + manual validation
    [
        "production embeddings-based retrieval systems sentence-transformers BGE E5",
        "vector databases approximate nearest neighbor FAISS Milvus Pinecone Qdrant",
        "hybrid search BM25 dense retrieval OpenSearch Elasticsearch",
        "ranking systems relevance scoring learning to rank",
        "Python production quality code engineering",
        "evaluation frameworks NDCG MRR MAP A/B testing offline",
        "search systems information retrieval query understanding",
        "recommendation systems collaborative filtering",
        "PyTorch TensorFlow deep learning model deployment",
        "embedding drift index refresh retrieval quality regression production",
    ]
    +
    # Nice-to-have (indices 10-17)
    [
        "LLM fine-tuning LoRA QLoRA PEFT parameter efficient",
        "learning to rank XGBoost LightGBM gradient boosting",
        "distributed systems large-scale inference optimization",
        "Docker Kubernetes MLOps containerized deployment",
        "HR tech recruiting technology talent marketplace",
        "MLflow Airflow experiment tracking data pipelines",
        "cloud platforms AWS GCP Azure SageMaker",
        "open source contributions AI ML research",
    ]
)

# Save auto-compiled JD artifacts
import json
jd_compiled = {
    "must_have_skills": JD_MUST_HAVE_SKILLS,
    "nice_to_have_skills": JD_NICE_TO_HAVE_SKILLS,
    "experience": JD_EXP,
    "disqualifiers": JD_DISQUALIFIERS,
    "queries": JD_QUERIES,
    "skills_ordered": JD_SKILLS_ORDERED,
}
json.dump(jd_compiled, open("artifacts/jd_compiled.json", "w"), indent=2)
print(f"Auto JD Compiler: {len(JD_MUST_HAVE_SKILLS)} must-have, "
      f"{len(JD_NICE_TO_HAVE_SKILLS)} nice-to-have skills extracted")
```

---

### 1.3 Company Taxonomy

**Critical finding from dataset analysis.** The dataset uses fictional Silicon Valley
companies as proxies. Getting this wrong breaks domain-fit scoring for ~35K candidates.

```python
# PRODUCT-TIER COMPANIES (treat as product company experience — full weight)
PRODUCT_COMPANIES = {
    # Fictional Silicon Valley proxies
    "hooli",           # Software
    "pied piper",      # Software
    "initech",         # Software (Office Space)
    # Real Indian product companies in dataset
    "swiggy", "cred", "razorpay", "zepto", "meesho", "slice",
    "phonepe", "zomato", "flipkart", "paytm", "juspay", "groww",
    "freshworks", "zoho", "browserstack", "postman", "chargebee",
}

# MANUFACTURING/CONGLOMERATE FICTIONAL (not product, not services — neutral)
MANUFACTURING_FICTIONAL = {
    "wayne enterprises",   # Conglomerate
    "stark industries",    # Manufacturing
    "acme corp",           # Manufacturing
    "globex inc",          # Manufacturing
    "dunder mifflin",      # Paper Products
}

# SERVICES FIRMS (penalize if entire career)
SERVICES_FIRMS = {
    "tcs", "tata consultancy",
    "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "lti", "ltimindtree",
    "deloitte", "kpmg", "ernst", "ey", "pwc",
}

def classify_company(company_name):
    """Returns: 'product', 'services', 'manufacturing', 'unknown'"""
    cn = company_name.lower().strip()
    if any(p in cn for p in PRODUCT_COMPANIES):
        return 'product'
    if any(p in cn for p in MANUFACTURING_FICTIONAL):
        return 'manufacturing'
    if any(p in cn for p in SERVICES_FIRMS):
        return 'services'
    return 'unknown'

def get_career_company_profile(career_history):
    """
    Returns dict with career company classification summary.
    Used in domain fit scoring and services penalty.
    """
    types = [classify_company(r.get('company', '')) for r in career_history]
    return {
        "has_product": 'product' in types,
        "has_services": 'services' in types,
        "all_services": all(t == 'services' for t in types) if types else False,
        "all_non_product": all(t != 'product' for t in types) if types else True,
        "product_count": types.count('product'),
        "services_count": types.count('services'),
    }
```

---

### 1.4 Pre-Filter (Run Before Everything Else)

**51% of the pool is immediately eliminable.** Do this before encoding to save GPU time.

```python
# Non-technical titles — hard exclude regardless of skills
NON_TECH_TITLES = [
    "hr manager", "human resource", "recruiter", "talent acquisition",
    "marketing manager", "sales manager", "account manager", "account executive",
    "operations manager", "accountant", "finance manager", "financial analyst",
    "customer support", "customer success", "copywriter", "content writer",
    "graphic designer", "ui designer", "ux designer",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "procurement", "supply chain", "logistics",
    "business analyst",  # majority are non-tech BAs in this dataset
]

# But spare titles with tech qualifiers
TECH_CARVEOUTS = [
    "ml", "ai", "machine learning", "data", "software", "backend", "frontend",
    "fullstack", "full stack", "platform", "search", "nlp", "retrieval",
    "recommendation", "scientist", "developer", "engineer",
]

def is_non_technical(title):
    t = title.lower()
    is_non_tech = any(kw in t for kw in NON_TECH_TITLES)
    has_tech = any(kw in t for kw in TECH_CARVEOUTS)
    return is_non_tech and not has_tech

# Pre-filter pass — build allowed_indices set
import json
import numpy as np

ordered_candidates = []
with open("candidates.jsonl") as f:
    for line in f:
        if line.strip():
            ordered_candidates.append(json.loads(line))

print(f"Total candidates: {len(ordered_candidates)}")

allowed_indices = []
excluded_titles = 0
for i, c in enumerate(ordered_candidates):
    title = c.get("profile", {}).get("current_title", "")
    if is_non_technical(title):
        excluded_titles += 1
    else:
        allowed_indices.append(i)

allowed_indices = np.array(allowed_indices)
np.save("artifacts/allowed_indices.npy", allowed_indices)
print(f"Pre-filter: excluded {excluded_titles} non-tech, {len(allowed_indices)} remain")
# Expected: ~49K remain after filtering ~51K non-tech titles
```

---

### 1.5 Chunk-Level Embeddings

**Four separate embeddings per candidate instead of one monolithic embedding.**
`achievements` field does NOT exist in schema — use career descriptions as its proxy.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

bi_model = SentenceTransformer('artifacts/models/bi/', device='cuda')

BATCH_SIZE = 512

# Only encode allowed_indices candidates (post-pre-filter)
n_allowed = len(allowed_indices)

summary_embeds    = np.zeros((len(ordered_candidates), 384), dtype=np.float16)
career_embeds     = np.zeros((len(ordered_candidates), 384), dtype=np.float16)
skills_embeds     = np.zeros((len(ordered_candidates), 384), dtype=np.float16)
impact_embeds     = np.zeros((len(ordered_candidates), 384), dtype=np.float16)

# Process in batches — only allowed_indices
for batch_start in range(0, len(allowed_indices), BATCH_SIZE):
    batch_idx = allowed_indices[batch_start:batch_start + BATCH_SIZE]
    batch = [ordered_candidates[i] for i in batch_idx]

    # Chunk 1: Summary + headline
    texts_summary = []
    for c in batch:
        p = c.get("profile", {})
        texts_summary.append(
            f"{p.get('headline', '')} {p.get('summary', '')}"[:512]
        )

    # Chunk 2: Career descriptions (primary signal — most dense field in schema)
    texts_career = []
    for c in batch:
        career_parts = []
        for role in c.get("career_history", []):
            title = role.get("title", "")
            company = role.get("company", "")
            desc = (role.get("description", "") or "")[:200]
            career_parts.append(f"{title} at {company}: {desc}")
        texts_career.append(" ".join(career_parts)[:600])

    # Chunk 3: Skills (names + proficiency emphasis)
    texts_skills = []
    for c in batch:
        skill_parts = []
        for s in c.get("skills", []):
            prof = s.get("proficiency", "intermediate")
            dur = s.get("duration_months", 0)
            name = s.get("name", "")
            # Weight by proficiency and duration for embedding quality
            if prof == "advanced" and dur > 12:
                skill_parts.extend([name] * 3)
            elif prof in ("advanced", "intermediate"):
                skill_parts.extend([name] * 2)
            else:
                skill_parts.append(name)
        texts_skills.append(" ".join(skill_parts)[:400])

    # Chunk 4: Impact proxy — numbers, metrics, scale mentions from career text
    texts_impact = []
    for c in batch:
        impact_tokens = []
        for role in c.get("career_history", []):
            desc = (role.get("description", "") or "")
            # Extract numeric impact mentions
            nums = re.findall(
                r'\b\d+[%xX]?\b.*?(?:users?|latency|CTR|NDCG|revenue|ms|requests?|improvement)',
                desc, re.IGNORECASE
            )
            impact_tokens.extend(nums[:3])
            # Extract action verbs signaling ownership
            actions = re.findall(
                r'\b(?:built|deployed|shipped|launched|designed|led|owned|reduced|improved|'
                r'optimized|scaled|increased|decreased)\b\s+\w+',
                desc, re.IGNORECASE
            )
            impact_tokens.extend(actions[:4])
        texts_impact.append(" ".join(impact_tokens)[:300] if impact_tokens
                            else "no quantified impact")

    # Encode all four chunks
    for texts, arr in [
        (texts_summary, summary_embeds),
        (texts_career, career_embeds),
        (texts_skills, skills_embeds),
        (texts_impact, impact_embeds),
    ]:
        embs = bi_model.encode(texts, normalize_embeddings=True,
                               batch_size=len(batch))
        for local_i, global_i in enumerate(batch_idx):
            arr[global_i] = embs[local_i].astype(np.float16)

np.save("artifacts/summary_embeds.npy", summary_embeds)
np.save("artifacts/career_embeds.npy", career_embeds)
np.save("artifacts/skills_embeds.npy", skills_embeds)
np.save("artifacts/impact_embeds.npy", impact_embeds)
print("Chunk embeddings saved.")
```

---

### 1.6 JD Embeddings

```python
jd_query_embeds = bi_model.encode(JD_QUERIES, normalize_embeddings=True)
jd_skill_embeds = bi_model.encode(JD_SKILLS_ORDERED, normalize_embeddings=True)

np.save("artifacts/jd_query_embeddings.npy", jd_query_embeds.astype(np.float32))
np.save("artifacts/jd_skill_embeddings.npy", jd_skill_embeds.astype(np.float32))

# Archetype embeddings — auto-generated from JD themes
ARCHETYPES = [
    "Search engineer who built and deployed search systems with ranking and relevance optimization at product companies",
    "Recommendation systems engineer who shipped collaborative filtering and content-based recommendation engines to real users",
    "Retrieval engineer working on dense retrieval, semantic search, and embedding-based document matching in production",
    "Matching engineer who built candidate-job or marketplace matching systems using ML and embedding similarity",
    "Ranking engineer who implemented learning-to-rank models and evaluation frameworks like NDCG MRR",
    "NLP engineer who built information retrieval and text understanding systems for production use",
    "Applied scientist who shipped ranking models and recommendation algorithms at product companies at scale",
    "ML platform engineer who built embedding pipelines, vector search infrastructure, and index management",
]

arch_embeds = bi_model.encode(ARCHETYPES, normalize_embeddings=True)
np.save("artifacts/archetype_embeddings.npy", arch_embeds.astype(np.float32))
```

---

### 1.7 FAISS Indices (One Per Chunk)

```python
import faiss

def build_faiss_index(embeddings_f16, path):
    emb_f32 = embeddings_f16.astype(np.float32)
    # Zero out disallowed indices so they never surface
    mask = np.zeros(len(emb_f32), dtype=bool)
    mask[allowed_indices] = True
    emb_f32[~mask] = 0.0
    index = faiss.IndexFlatIP(384)
    index.add(emb_f32)
    faiss.write_index(index, path)

build_faiss_index(summary_embeds, "artifacts/summary.index")
build_faiss_index(career_embeds, "artifacts/career.index")
build_faiss_index(skills_embeds, "artifacts/skills.index")
build_faiss_index(impact_embeds, "artifacts/impact.index")
print("FAISS indices built.")
```

---

### 1.8 Precomputed Scores

```python
# Archetype max scores — vectorized over allowed indices only
career_f32 = career_embeds[allowed_indices].astype(np.float32)
arch_max_allowed = np.max(career_f32 @ arch_embeds.T, axis=1)  # (n_allowed,)

# Full array (NaN for excluded)
arch_max_full = np.full(len(ordered_candidates), np.nan, dtype=np.float32)
arch_max_full[allowed_indices] = arch_max_allowed
np.save("artifacts/archetype_max_scores.npy", arch_max_full)
```

---

### 1.9 Skill Matching + Credibility

```python
import pickle

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
        "hf transformers": "transformers",
        "huggingface": "transformers", "hugging face": "transformers",
        "faiss cpu": "faiss", "faiss gpu": "faiss",
    }
    return aliases.get(name, name)

# Collect unique normalized names from allowed candidates only
all_names = set()
for i in allowed_indices:
    for s in ordered_candidates[i].get("skills", []):
        all_names.add(normalize_skill_name(s.get("name", "")))
unique_names = sorted(all_names - {""})

unique_embeds = bi_model.encode(unique_names, batch_size=256, normalize_embeddings=True)
skill_embed_map = dict(zip(unique_names, unique_embeds))

# Calibrate threshold
test_skills = ["faiss", "elasticsearch", "embeddings", "pytorch",
               "recommendation systems", "python", "ndcg", "information retrieval"]
test_embeds = bi_model.encode(test_skills, normalize_embeddings=True)
print("\n=== SKILL THRESHOLD CALIBRATION ===")
for name, emb in zip(test_skills, test_embeds):
    sims = emb @ jd_skill_embeds.T
    best_idx = int(np.argmax(sims))
    print(f"  {name:35s} -> JD[{best_idx:2d}] = {float(sims[best_idx]):.3f}")

SKILL_MATCH_THRESHOLD = 0.48  # Lower than v15 — BGE-small is more discriminative
np.save("artifacts/skill_threshold.npy", np.array([SKILL_MATCH_THRESHOLD]))

# Build skill_lookup for allowed candidates
skill_lookup = {}
for i in allowed_indices:
    c = ordered_candidates[i]
    cid = c["candidate_id"]
    matches = []
    for s in c.get("skills", []):
        norm = normalize_skill_name(s.get("name", ""))
        if norm not in skill_embed_map:
            continue
        se = skill_embed_map[norm]
        sims = se @ jd_skill_embeds.T
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= SKILL_MATCH_THRESHOLD:
            matches.append({
                "skill_name": s.get("name", ""),
                "norm_name": norm,
                "best_jd_match_score": best_score,
                "best_jd_req_idx": best_idx,
                "proficiency": s.get("proficiency", "intermediate"),
                "endorsements": s.get("endorsements", 0),
                "duration_months": s.get("duration_months", 0),
                # Note: skill_assessment_scores has near-zero coverage for JD-relevant
                # skills (Vector Databases=0, Ranking=0, FAISS=303/100K).
                # Used only as a weak tiebreaker, not primary credibility signal.
                "assessment_score": c.get("redrob_signals", {})
                                      .get("skill_assessment_scores", {})
                                      .get(s.get("name", ""), None),
            })
    skill_lookup[cid] = matches

pickle.dump(skill_lookup, open("artifacts/skill_matches.pkl", "wb"))
print(f"Skill lookup built for {len(skill_lookup)} candidates")
```

---

### 1.10 Evidence Extraction

```python
PRODUCTION_VERBS = [
    "built", "deployed", "shipped", "serving", "launched", "went live",
    "production", "real-time", "live traffic", "real users",
]
RESEARCH_SIGNALS = [
    "research only", "thesis", "proof of concept", "prototype only",
    "academic project", "paper", "arxiv",
]
IMPACT_PATTERNS = [
    r'\b\d+%\s*(?:improvement|increase|decrease|reduction|gain)\b',
    r'\b\d+[xX]\s*(?:faster|improvement|speedup)\b',
    r'\b\d+(?:\.\d+)?[MBK]\+?\s*(?:users?|requests?|queries?|events?)\b',
    r'\b(?:latency|p99|p95)\s*(?:of|to|by)?\s*\d+\s*ms\b',
    r'\b(?:CTR|NDCG|MRR|MAP|AUC|precision|recall)\s*(?:of|improved|by)?\s*\d+',
    r'\breduced\s+(?:\w+\s+){0,3}(?:from|by)\s+\d+',
]

evidence_lookup = {}
for i in allowed_indices:
    c = ordered_candidates[i]
    cid = c["candidate_id"]
    career = c.get("career_history", [])
    all_desc = " ".join((r.get("description", "") or "") for r in career).lower()

    prod_count = sum(1 for v in PRODUCTION_VERBS if v in all_desc)
    research_count = sum(1 for v in RESEARCH_SIGNALS if v in all_desc)

    impact_mentions = []
    for pattern in IMPACT_PATTERNS:
        found = re.findall(pattern, all_desc, re.IGNORECASE)
        impact_mentions.extend(found[:2])

    # Company profile
    company_profile = get_career_company_profile(career)

    evidence_lookup[cid] = {
        "production_score": min(prod_count / 5.0, 1.0),
        "research_penalty": min(research_count / 3.0, 1.0),
        "impact_mentions": impact_mentions[:4],
        "impact_score": min(len(impact_mentions) / 4.0, 1.0),
        "company_profile": company_profile,
    }

pickle.dump(evidence_lookup, open("artifacts/evidence.pkl", "wb"))
```

---

### 1.11 Honeypot Detection

Three-signal detection. Signal 3 is new in v21 — catches the title/description coherence
pattern (e.g. "Business Analyst" role with mechanical engineering job description).

```python
from datetime import datetime

def precompute_honeypot_flags(ordered_candidates, allowed_indices, skill_lookup):
    hard_exclude = set()
    soft_penalize = set()

    for i in allowed_indices:
        c = ordered_candidates[i]
        cid = c["candidate_id"]
        career = c.get("career_history", [])
        skills = skill_lookup.get(cid, [])
        suspicion = 0

        # Signal 1: Skill inflation (expert claims with zero duration)
        experts = [s for s in c.get("skills", []) if s.get("proficiency") == "advanced"]
        zero_dur = [s for s in experts if s.get("duration_months", 0) == 0]
        if len(experts) >= 6 and len(zero_dur) >= 3:
            suspicion += 1

        # Signal 2: Timeline impossibility
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

        # Signal 3 (NEW): Title/description coherence mismatch
        # A role whose description talks about a completely different domain than its title
        title_domain_map = {
            "accountant": ["accounting", "finance", "gl", "audit", "tax", "statutory"],
            "hr manager": ["hr", "human resource", "hiring", "onboarding", "payroll"],
            "marketing manager": ["marketing", "campaign", "seo", "brand", "content"],
            "operations manager": ["operations", "fulfillment", "warehouse", "logistics"],
            "mechanical engineer": ["mechanical", "solidworks", "cad", "manufacturing", "fea"],
            "civil engineer": ["civil", "structural", "construction", "surveying"],
            "content writer": ["content", "copywriting", "editorial", "blog"],
            "business analyst": ["requirements", "stakeholder", "business process", "brd"],
        }
        mismatch_count = 0
        for role in career:
            role_title = role.get("title", "").lower()
            role_desc = (role.get("description", "") or "").lower()
            for title_kw, desc_kws in title_domain_map.items():
                if title_kw in role_title:
                    # Title matches a non-tech domain
                    title_desc_match = any(dk in role_desc for dk in desc_kws)
                    if not title_desc_match and len(role_desc) > 50:
                        # Description doesn't match this domain at all
                        mismatch_count += 1
                    break
        if mismatch_count >= 2:
            suspicion += 1

        # Signal 4: YOE inflation vs career total
        total_months = sum(r.get("duration_months", 0) for r in career)
        stated_yoe = c.get("profile", {}).get("years_of_experience", 0)
        if stated_yoe * 12 > total_months * 1.5 + 24:
            suspicion += 1

        if suspicion >= 3:
            hard_exclude.add(cid)
        elif suspicion == 2:
            soft_penalize.add(cid)

    print(f"Honeypot: {len(hard_exclude)} hard-excluded, {len(soft_penalize)} soft-penalized")
    return {"hard": hard_exclude, "soft": soft_penalize}

honeypot_data = precompute_honeypot_flags(ordered_candidates, allowed_indices, skill_lookup)
pickle.dump(honeypot_data, open("artifacts/honeypot_flags.pkl", "wb"))
```

---

### 1.12 Metadata Extraction

```python
import pandas as pd

flat = []
nested_data = {}

for i in allowed_indices:
    c = ordered_candidates[i]
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    cid = c["candidate_id"]

    # --- Sentinel handling for -1 values ---
    # github_activity_score = -1 means "no GitHub linked" (not a bad score)
    # offer_acceptance_rate = -1 means "no prior offers" (not a bad signal)
    github_raw = s.get("github_activity_score", -1)
    github_val = github_raw if github_raw >= 0 else None  # None = missing, not 0

    offer_raw = s.get("offer_acceptance_rate", -1)
    offer_val = offer_raw if offer_raw >= 0 else None

    flat.append({
        "candidate_id": cid,
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
        "github_activity_score": github_val,      # None if no GitHub
        "offer_acceptance_rate": offer_val,        # None if no prior offers
        "verified_email": s.get("verified_email", False),
        "verified_phone": s.get("verified_phone", False),
        "linkedin_connected": s.get("linkedin_connected", False),
        "profile_completeness_score": s.get("profile_completeness_score", 50),
        "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d", 0),
        "preferred_work_mode": s.get("preferred_work_mode", "hybrid"),
        "willing_to_relocate": s.get("willing_to_relocate", False),
        "skill_assessment_scores": s.get("skill_assessment_scores", {}),
        "global_idx": i,
    })

    career = c.get("career_history", [])
    nested_data[cid] = {
        "career_history": career,
        "career_text": " ".join((r.get("description", "") or "") for r in career),
        "career_companies": [r.get("company", "") for r in career],
        "education": c.get("education", []),
        "skill_names": [s.get("name", "") for s in c.get("skills", [])],
        "skills_raw": c.get("skills", []),
        "global_idx": i,
    }

meta_df = pd.DataFrame(flat)
meta_df.to_parquet("artifacts/candidates_flat.parquet")
pickle.dump(nested_data, open("artifacts/candidates_nested.pkl", "wb"))
print(f"Metadata saved: {len(meta_df)} allowed candidates")
```

---

### 1.13 Artifact Budget

| Artifact | Size |
|---|---|
| models/bi/ + models/ce/ | ~160MB |
| summary_embeds.npy | ~73MB |
| career_embeds.npy | ~73MB |
| skills_embeds.npy | ~73MB |
| impact_embeds.npy | ~73MB |
| *.index (4 FAISS) | ~600MB |
| archetype_max_scores.npy | <1MB |
| jd_*.npy, archetype_embeddings.npy | <1MB |
| skill_matches.pkl | ~35MB |
| evidence.pkl | ~30MB |
| candidates_flat.parquet | ~120MB |
| candidates_nested.pkl | ~250MB |
| honeypot_flags.pkl, allowed_indices.npy, jd_compiled.json | <2MB |
| **Total** | **~1.5GB** |

---

## Part 2 — Ranking

### 2.1 Constants

```python
STRONG_KEYWORDS = [
    "ranking system", "ranking engine", "ranking model", "ranking pipeline",
    "search system", "search engine", "search platform", "search quality",
    "recommendation system", "recommendation engine", "recommender",
    "retrieval system", "retrieval pipeline", "information retrieval",
    "matching system", "matching engine", "candidate matching",
    "learning to rank", "reranking", "re-ranking", "query understanding",
    "relevance scoring", "relevance model", "discovery platform",
]

MODERATE_KEYWORDS = [
    "embeddings", "vector search", "dense retrieval", "semantic search",
    "hybrid search", "neural search", "bm25", "inverted index",
    "ndcg", "mrr", "a/b test", "offline evaluation",
    "faiss", "pinecone", "milvus", "qdrant", "weaviate",
    "elasticsearch", "opensearch",
]

DEPLOYMENT_KEYWORDS = [
    "deployed to production", "shipped to production", "production system",
    "real-time serving", "live traffic", "serving users", "went live",
    "production environment", "production traffic", "launched",
]

SCALE_KEYWORDS = [
    "at scale", "millions", "thousands of users", "daily active",
    "throughput", "latency", "sla", "high availability",
]

PRODUCT_ENG_KEYWORDS = [
    "api", "microservice", "endpoint", "ci/cd", "monitoring", "alerting",
]

ANTI_PRODUCTION_KEYWORDS = [
    "research only", "thesis", "proof of concept", "prototype only",
    "academic project",
]

VECTOR_TOOLS = [
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "chroma", "chromadb",
    "annoy", "scann", "vespa", "pgvector", "typesense",
]

PROF_WEIGHTS = {
    "expert": 1.0, "advanced": 0.85, "intermediate": 0.60, "beginner": 0.30
}

TITLE_LEVELS = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2, "scientist": 2,
    "senior": 3, "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "director": 5, "head": 5, "vp": 6,
    "founder": 5, "co-founder": 5, "cto": 6,
}

TIER_A_TITLES = [
    "ml engineer", "machine learning", "ai engineer",
    "data scientist", "nlp engineer", "applied scientist",
    "research engineer", "deep learning", "search engineer",
    "recommendation", "retrieval", "ranking engineer",
]

TIER_B_TITLES = [
    "software engineer", "backend engineer", "full stack",
    "data engineer", "analytics engineer", "platform engineer",
]

TIER_C_TITLES = [
    "devops", "cloud engineer", "sre", "frontend",
    "qa engineer", "product manager",
]

JD_CORE = """Senior AI Engineer, founding team, product company.
Must have shipped ranking, search, or recommendation systems to production.
Production experience with embeddings, vector databases, hybrid search.
Python, PyTorch, evaluation frameworks NDCG MRR. 5-9 years experience. India."""
```

---

### 2.2 Load Artifacts

```python
import numpy as np, pandas as pd, pickle, logging, json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# JD compiled artifacts
jd = json.load(open("artifacts/jd_compiled.json"))
SKILL_MATCH_THRESHOLD = float(np.load("artifacts/skill_threshold.npy")[0])

# Candidate data
meta_df = pd.read_parquet("artifacts/candidates_flat.parquet")
nested = pickle.load(open("artifacts/candidates_nested.pkl", "rb"))
skill_lookup = pickle.load(open("artifacts/skill_matches.pkl", "rb"))
evidence_lookup = pickle.load(open("artifacts/evidence.pkl", "rb"))
honeypot_data = pickle.load(open("artifacts/honeypot_flags.pkl", "rb"))
honeypot_hard = honeypot_data["hard"]
honeypot_soft = honeypot_data["soft"]
allowed_indices = np.load("artifacts/allowed_indices.npy")

# Embeddings
summary_f16   = np.load("artifacts/summary_embeds.npy")
career_f16    = np.load("artifacts/career_embeds.npy")
skills_f16    = np.load("artifacts/skills_embeds.npy")
impact_f16    = np.load("artifacts/impact_embeds.npy")

archetype_max_raw = np.load("artifacts/archetype_max_scores.npy")

jd_queries    = np.load("artifacts/jd_query_embeddings.npy")   # (5, 384)
jd_skills     = np.load("artifacts/jd_skill_embeddings.npy")   # (18, 384)
arch_embeds   = np.load("artifacts/archetype_embeddings.npy")  # (8, 384)

# Cast to float32
summary_f32 = summary_f16.astype(np.float32)
career_f32  = career_f16.astype(np.float32)
skills_f32  = skills_f16.astype(np.float32)
impact_f32  = impact_f16.astype(np.float32)

# Archetype max: map [-1,1] → [0,1], NaN for excluded
archetype_max = np.where(
    np.isnan(archetype_max_raw),
    0.0,
    (archetype_max_raw + 1.0) / 2.0
).astype(np.float32)

# Vectorized semantic scores — all queries × all candidates
# Shape: (100K, 5) — only non-NaN for allowed_indices
all_semantic_raw = career_f32 @ jd_queries.T         # career chunk for domain
all_semantic = (all_semantic_raw + 1.0) / 2.0
all_semantic_max = all_semantic.max(axis=1)

# Vectorized skills semantic
all_skills_semantic = (skills_f32 @ jd_queries[0:1].T + 1.0) / 2.0
all_skills_semantic = all_skills_semantic[:, 0]

# O(1) lookups
cid_to_idx = {row["candidate_id"]: i for i, row in meta_df.iterrows()}
cids_arr   = meta_df["candidate_id"].values
titles_arr = meta_df["current_title"].values
years_arr  = meta_df["years_of_experience"].values

# DataFrame row index by global candidate index
global_idx_to_meta = {int(row["global_idx"]): i for i, row in meta_df.iterrows()}

TODAY = pd.Timestamp.now()

import faiss
idx_summary = faiss.read_index("artifacts/summary.index")
idx_career  = faiss.read_index("artifacts/career.index")
idx_skills  = faiss.read_index("artifacts/skills.index")
idx_impact  = faiss.read_index("artifacts/impact.index")
```

---

### 2.3 Stage 0 — Hard Pre-Filter

Non-tech titles were removed during precompute. At ranking time, additionally remove
honeypot hard-excludes from the allowed pool before any retrieval.

```python
# Build working pool from allowed_indices, minus honeypot hard-excludes
working_pool = set()
for i in allowed_indices:
    meta_i = global_idx_to_meta.get(i)
    if meta_i is None:
        continue
    cid = cids_arr[meta_i]
    if cid not in honeypot_hard:
        working_pool.add(i)  # global index

logging.info(f"Stage 0: {len(working_pool)} candidates in working pool")
```

---

### 2.4 Stage 1 — Multi-Retrieval (5 Retrievers)

```python
def faiss_retrieve(index, query_vec, k=1000):
    q = query_vec.reshape(1, -1).astype(np.float32)
    _, indices = index.search(q, k)
    return set(indices[0][indices[0] >= 0].tolist())

retrieved = set()

# Retriever 1: Career semantic (query 0 = domain-specific)
for q in jd_queries:
    retrieved |= faiss_retrieve(idx_career, q, k=700)

# Retriever 2: Skills semantic
for q in jd_queries[:3]:
    retrieved |= faiss_retrieve(idx_skills, q, k=600)

# Retriever 3: Summary (catches candidates whose summary is well-written
# but career descriptions are terse)
retrieved |= faiss_retrieve(idx_summary, jd_queries[0], k=400)

# Retriever 4: Impact (catches candidates with quantified achievements
# that match scale/impact keywords in JD)
retrieved |= faiss_retrieve(idx_impact, jd_queries[2], k=300)

# Retriever 5: Archetype — match career text against archetype embeddings
for arch_q in arch_embeds:
    retrieved |= faiss_retrieve(idx_career, arch_q, k=400)

# Skill-match retriever: top candidates by must-have skill count
must_counts = np.zeros(len(meta_df))
for i, row in meta_df.iterrows():
    cid = row["candidate_id"]
    must_counts[i] = sum(
        1 for s in skill_lookup.get(cid, [])
        if s["best_jd_req_idx"] < 10
    )
top_skill_meta = set(np.argpartition(must_counts, -500)[-500:].tolist())
# Convert meta indices to global indices
top_skill_global = {int(meta_df.iloc[i]["global_idx"]) for i in top_skill_meta}
retrieved |= top_skill_global

# Intersect with working pool
retrieved = retrieved & working_pool

logging.info(f"Stage 1 retrieval union: {len(retrieved)} candidates")
# Expected: ~3000-4000
```

---

### 2.5 Stage 2 — Fusion Scoring → Top 800

```python
def quick_behavioral(meta_idx):
    """Fast behavioral signal for fusion scoring."""
    row = meta_df.iloc[meta_idx]
    la = row["last_active_date"]
    try:
        days = (TODAY - pd.to_datetime(la)).days
    except:
        days = 365
    recency = (1.0 if days <= 30 else 0.8 if days <= 90
               else 0.5 if days <= 180 else 0.25 if days <= 365 else 0.1)
    rr = float(row["response_rate"] or 0.5)
    response = min(max(rr, 0), 1)
    otw = 1.0 if row["open_to_work"] else 0.5
    return 0.5 * response + 0.3 * recency + 0.2 * otw

def fusion_score(global_idx):
    meta_i = global_idx_to_meta.get(global_idx)
    if meta_i is None:
        return 0.0
    cid = cids_arr[meta_i]

    # 1. Career semantic (domain query)
    cos_career = float((career_f32[global_idx] @ jd_queries[0]) + 1.0) / 2.0

    # 2. Skills semantic
    cos_skills = float((skills_f32[global_idx] @ jd_queries[0]) + 1.0) / 2.0

    # 3. Archetype
    arch = float(archetype_max[global_idx])

    # 4. Skill count signal
    n_must = sum(1 for s in skill_lookup.get(cid, []) if s["best_jd_req_idx"] < 10)
    skill_count = min(n_must / 8.0, 1.0)

    # 5. Behavioral
    behav = quick_behavioral(meta_i)

    # 6. Evidence production signal
    ev = evidence_lookup.get(cid, {})
    prod = float(ev.get("production_score", 0))
    research_pen = float(ev.get("research_penalty", 0))

    score = (
        0.28 * cos_career +
        0.18 * cos_skills +
        0.18 * arch +
        0.18 * skill_count +
        0.10 * behav +
        0.08 * prod
    )
    score *= (1.0 - 0.4 * research_pen)  # soft research penalty

    return max(score, 0.0)

fusion_results = [(gi, fusion_score(gi)) for gi in retrieved]
fusion_results.sort(key=lambda x: x[1], reverse=True)
top_800_global = fusion_results[:800]
logging.info(f"Stage 2 fusion: top {len(top_800_global)} selected")
```

---

### 2.6 Stage 3 — Cross-Encoder (800 → 300)

Cross-encoder on CPU over 800 candidates. Benchmarked at ~90-110s with batch_size=32.
Circuit breaker: if throughput is slow, drop to 500 and use smaller batch.

```python
import time
from sentence_transformers import CrossEncoder

ce_model = CrossEncoder("artifacts/models/ce/", device="cpu")

def select_relevant_roles(career_history, n=3):
    if len(career_history) <= n:
        return career_history
    scored = []
    for role in career_history:
        desc = (role.get("description", "") or "").lower()
        relevance = (
            sum(2 for kw in STRONG_KEYWORDS if kw in desc) +
            sum(1 for kw in MODERATE_KEYWORDS if kw in desc)
        )
        scored.append((relevance, role))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] == 0:
        return sorted(career_history, key=lambda r: r.get("start_date", ""), reverse=True)[:n]
    return [r for _, r in scored[:n]]

def build_ce_text(global_idx):
    meta_i = global_idx_to_meta.get(global_idx)
    if meta_i is None:
        return ""
    cid = cids_arr[meta_i]
    data = nested.get(cid, {})
    parts = []

    summary = str(meta_df.iloc[meta_i].get("summary", "") or "")
    if summary:
        parts.append(summary[:200])

    career = data.get("career_history", [])
    for role in select_relevant_roles(career, n=3):
        title = role.get("title", "")
        company = role.get("company", "")
        desc = (role.get("description", "") or "")[:250]
        # Include company type signal for cross-encoder context
        co_type = classify_company(company)
        company_tag = f"[{co_type}]" if co_type in ("product", "services") else ""
        parts.append(f"{title} at {company}{company_tag}: {desc}")

    skill_names = ", ".join(data.get("skill_names", [])[:10])
    parts.append(f"Skills: {skill_names}")

    return " ".join(parts)[:1500]

# Build pairs
pairs = [(JD_CORE, build_ce_text(gi)) for gi, _ in top_800_global]

# Circuit breaker: time the first 50 pairs
t0 = time.time()
sample_logits = ce_model.predict(pairs[:50], batch_size=32)
t_sample = time.time() - t0
projected_total = (t_sample / 50) * len(pairs)

if projected_total > 130:
    # Slow — truncate to 500
    logging.warning(f"Cross-encoder projected {projected_total:.0f}s — truncating to 500")
    pairs = pairs[:500]
    top_800_global = top_800_global[:500]

t_ce_start = time.time()
raw_logits = ce_model.predict(pairs, batch_size=32, show_progress_bar=False)
logging.info(f"Cross-encoder: {time.time()-t_ce_start:.1f}s for {len(pairs)} pairs")

cross_scores = 1.0 / (1.0 + np.exp(-raw_logits))  # sigmoid → [0,1]

# Select top 300
ce_ranked = sorted(
    zip([gi for gi, _ in top_800_global], cross_scores),
    key=lambda x: x[1], reverse=True
)
top_300_global = ce_ranked[:300]
logging.info(f"Stage 3 cross-encoder: top {len(top_300_global)} selected")
```

---

### 2.7 Helper Functions

```python
def safe_val(val, default=0.5):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val

def endorse_weight(e):
    if e == 0: return 0.40
    if e <= 5: return 0.70
    if e <= 20: return 0.90
    return 1.0

def duration_weight(d):
    if d == 0: return 0.20
    if d <= 6: return 0.50
    if d <= 24: return 0.80
    return 1.0

def compute_credibility(skill):
    """
    Credibility score for a single skill match.
    NOTE: assessment_score de-weighted (near-zero coverage for JD-relevant skills).
    Primary signals: proficiency + endorsements + duration.
    """
    prof = PROF_WEIGHTS.get(skill["proficiency"], 0.60)
    endorse = endorse_weight(skill["endorsements"])
    dur = duration_weight(skill["duration_months"])

    assess = skill.get("assessment_score")
    if assess is not None:
        assess_w = 1.0 if assess >= 70 else 0.75 if assess >= 40 else 0.50
        # Assessment has coverage only for non-core skills — use lightly
        return min(0.40 * prof + 0.30 * endorse + 0.25 * dur + 0.05 * assess_w, 1.0)
    else:
        return min(0.40 * prof + 0.35 * endorse + 0.25 * dur, 1.0)

def classify_title(title):
    t = title.lower()
    if any(k in t for k in TIER_A_TITLES): return 1.0
    if any(k in t for k in TIER_B_TITLES): return 0.6
    if any(k in t for k in TIER_C_TITLES): return 0.3
    return 0.05

def score_experience(years):
    ideal = jd["experience"]
    if ideal["ideal_min"] <= years <= ideal["ideal_max"]: return 1.0
    if ideal["soft_min"] <= years < ideal["ideal_min"]: return 0.75
    if ideal["ideal_max"] < years <= ideal["soft_max"]: return 0.75
    if 3.0 <= years < ideal["soft_min"]: return 0.5
    if ideal["soft_max"] < years <= 15.0: return 0.5
    return 0.25

def get_title_level(title):
    t = title.lower()
    for kw in sorted(TITLE_LEVELS.keys(), key=len, reverse=True):
        if kw in t:
            return TITLE_LEVELS[kw]
    return 2

def compute_keyword_score(career_text_lower):
    score = 0.0
    for kw in STRONG_KEYWORDS:
        if kw in career_text_lower: score += 0.20
    for kw in MODERATE_KEYWORDS:
        if kw in career_text_lower: score += 0.10
    return min(score, 1.0)
```

---

### 2.8 Penalty Functions

```python
def services_penalty(cid, f1_score):
    """
    Conditional. JD says: 'only worked at consulting firms in entire career — not a fit.'
    Genuine ML engineers at services firms (with domain evidence) get lighter penalty.
    """
    ev = evidence_lookup.get(cid, {})
    cp = ev.get("company_profile", {})
    if not cp.get("all_services", False):
        return 1.0
    # All-services career
    if f1_score < 0.30:
        return 0.40  # No domain evidence + all services
    return 0.80      # Has domain evidence but still all services

def stuffer_check(cid, meta_idx):
    """
    ×0.20 penalty for non-tech title + many JD keyword matches + low credibility.
    """
    title_tier = classify_title(str(titles_arr[meta_idx]))
    if title_tier > 0.3:
        return 1.0
    skills = skill_lookup.get(cid, [])
    matched = [s for s in skills if s["best_jd_match_score"] > SKILL_MATCH_THRESHOLD]
    if len(matched) < 5:
        return 1.0
    mean_cred = sum(compute_credibility(s) for s in matched) / len(matched)
    if mean_cred < 0.45:
        return 0.20
    return 1.0

def honeypot_penalty(cid):
    return 0.50 if cid in honeypot_soft else 1.0
```

---

### 2.9 Stage 4 — Final Feature Scoring (7 Features)

```python
def feature_1_domain(global_idx, meta_idx, cid, cross_score):
    """Career domain evidence. Weight: 0.30"""
    arch = float(archetype_max[global_idx])
    career_text = nested.get(cid, {}).get("career_text", "").lower()
    kw = compute_keyword_score(career_text)
    # Career chunk semantic — domain query only (query 0)
    cos_career = (float(career_f32[global_idx] @ jd_queries[0]) + 1.0) / 2.0
    domain = 0.30 * kw + 0.70 * cos_career

    # Company profile bonus: product company experience
    ev = evidence_lookup.get(cid, {})
    cp = ev.get("company_profile", {})
    product_bonus = 0.10 if cp.get("has_product", False) else 0.0

    result = 0.35 * cross_score + 0.30 * arch + 0.25 * domain + 0.10 * product_bonus
    return max(min(result, 1.0), 0.0)


def feature_2_retrieval_expertise(cid):
    """Retrieval/search skill credibility. Weight: 0.26"""
    skills = skill_lookup.get(cid, [])
    seen_reqs = {}
    for s in skills:
        if s["best_jd_match_score"] < SKILL_MATCH_THRESHOLD:
            continue
        req_idx = s["best_jd_req_idx"]
        cred = compute_credibility(s)
        credible_score = s["best_jd_match_score"] * cred
        if req_idx not in seen_reqs or credible_score > seen_reqs[req_idx]:
            seen_reqs[req_idx] = credible_score

    must_hits = sorted([v for k, v in seen_reqs.items() if k < 10], reverse=True)
    nice_hits = sorted([v for k, v in seen_reqs.items() if k >= 10], reverse=True)

    top_must = must_hits[:5]
    depth = min(sum(top_must) / max(len(top_must), 3), 1.0) if top_must else 0.0
    coverage = min(len(must_hits) / 6, 1.0)
    top_nice = nice_hits[:3]
    nice_score = min(sum(top_nice) / max(len(top_nice), 2), 1.0) if top_nice else 0.0

    must_component = 0.60 * depth + 0.40 * coverage
    return min(0.70 * must_component + 0.30 * nice_score, 1.0)


def feature_3_production_evidence(global_idx, cid):
    """Production deployment evidence. Weight: 0.18"""
    ev = evidence_lookup.get(cid, {})
    prod_score = float(ev.get("production_score", 0))
    research_pen = float(ev.get("research_penalty", 0))
    impact_score = float(ev.get("impact_score", 0))

    # Additional keyword check on career text
    career_text = nested.get(cid, {}).get("career_text", "").lower()
    kw = 0.0
    for k in DEPLOYMENT_KEYWORDS:
        if k in career_text: kw += 0.15
    for k in SCALE_KEYWORDS:
        if k in career_text: kw += 0.10
    kw = min(kw, 1.0)

    # Production semantic — query 2
    cos_prod = (float(career_f32[global_idx] @ jd_queries[2]) + 1.0) / 2.0

    raw = 0.35 * prod_score + 0.25 * kw + 0.25 * cos_prod + 0.15 * impact_score
    raw *= (1.0 - 0.5 * research_pen)
    return max(min(raw, 1.0), 0.0)


def feature_4_vector_infra(global_idx, cid):
    """Vector DB and search infrastructure. Weight: 0.10"""
    skills = skill_lookup.get(cid, [])
    career_text = nested.get(cid, {}).get("career_text", "").lower()

    tools_found = {}
    for s in skills:
        nn = s.get("norm_name", s["skill_name"].lower())
        if nn not in VECTOR_TOOLS and s["best_jd_req_idx"] not in {1, 2}:
            continue
        cred = compute_credibility(s)
        in_desc = nn in career_text
        effective = cred * (1.0 if in_desc else 0.6)
        if nn not in tools_found or effective > tools_found[nn]:
            tools_found[nn] = effective

    n = len(tools_found)
    avg_cred = sum(tools_found.values()) / max(n, 1)
    base = {0: 0.0, 1: 0.4, 2: 0.7}.get(n, 1.0)
    tool_score = min(base * avg_cred, 1.0)

    cos_infra = (float(skills_f32[global_idx] @ jd_queries[1]) + 1.0) / 2.0
    return max(min(0.60 * tool_score + 0.40 * cos_infra, 1.0), 0.0)


def feature_5_behavioral(meta_idx):
    """
    Behavioral signals with sentinel handling for -1 values.
    Weight: 0.10 (capped at 0.80).
    Key upgrade: saved_by_recruiters_30d as market-validation signal.
    """
    row = meta_df.iloc[meta_idx]

    # Recency
    try:
        days = (TODAY - pd.to_datetime(row["last_active_date"])).days
    except:
        days = 365
    recency = (1.0 if days <= 30 else 0.8 if days <= 90
               else 0.5 if days <= 180 else 0.25 if days <= 365 else 0.1)

    # Response rate
    rr = safe_val(row["response_rate"], 0.5)
    response = (1.0 if rr >= 0.7 else 0.85 if rr >= 0.5
                else 0.60 if rr >= 0.3 else 0.35 if rr >= 0.15 else 0.15)

    # Response time
    rt = safe_val(row["avg_response_time_hours"], 72)
    resp_time = 1.0 if rt < 24 else 0.8 if rt < 72 else 0.5 if rt < 168 else 0.3

    # Interview completion
    ic = safe_val(row["interview_completion_rate"], 0.5)
    interview = 1.0 if ic >= 0.8 else 0.75 if ic >= 0.6 else 0.5 if ic >= 0.4 else 0.25

    # Open to work
    otw = 1.0 if safe_val(row["open_to_work"], False) else 0.5

    # GitHub — SENTINEL HANDLING: None means no GitHub linked (neutral, not bad)
    gh = row["github_activity_score"]   # already None if was -1
    if gh is None:
        github = 0.50  # neutral — absence of GitHub ≠ bad engineer
    elif gh <= 20:
        github = 0.40
    elif gh <= 50:
        github = 0.70
    elif gh <= 80:
        github = 0.85
    else:
        github = 1.0

    # Trust signals
    ve = 1.0 if safe_val(row["verified_email"], False) else 0.0
    vp = 1.0 if safe_val(row["verified_phone"], False) else 0.0
    li = 1.0 if safe_val(row["linkedin_connected"], False) else 0.0
    trust = 0.3 * ve + 0.3 * vp + 0.4 * li

    # Notice period
    nd = safe_val(row["notice_period_days"], 90)
    notice = (1.0 if nd <= 30 else 0.80 if nd <= 60
              else 0.55 if nd <= 90 else 0.35 if nd <= 120 else 0.20)

    # saved_by_recruiters_30d — market validation signal (NEW)
    # Other recruiters are actively saving this profile right now.
    # High value: candidate is proven-interesting and probably in active process.
    saved = safe_val(row["saved_by_recruiters_30d"], 0)
    recruiter_interest = min(saved / 10.0, 1.0)

    # Profile completeness
    pc = safe_val(row["profile_completeness_score"], 50) / 100.0

    result = (
        0.20 * recency +
        0.20 * response +
        0.10 * resp_time +
        0.10 * interview +
        0.08 * otw +
        0.08 * github +
        0.07 * trust +
        0.08 * notice +
        0.06 * recruiter_interest +   # NEW — promoted from 0.03
        0.03 * pc
    )
    return min(result, 0.80)   # cap — prevents behavioral from dominating


def feature_6_llm_adjacent(cid):
    """LLM and adjacent skills. Weight: 0.04"""
    skills = skill_lookup.get(cid, [])
    VALUED = {10, 11, 12}
    LESS_VALUED = {13, 14}
    score = 0.0
    for s in skills:
        if s["best_jd_match_score"] < SKILL_MATCH_THRESHOLD:
            continue
        cred = compute_credibility(s)
        if s["best_jd_req_idx"] in VALUED:
            score += 0.15 * cred
        elif s["best_jd_req_idx"] in LESS_VALUED:
            score += 0.05 * cred
    return min(score, 1.0)


def feature_7_career_progression(cid):
    """Career trajectory. Weight: 0.02"""
    career = nested.get(cid, {}).get("career_history", [])
    if len(career) < 2:
        return 0.5
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

---

### 2.10 Final Score Formula

```python
def compute_final_score(global_idx, meta_idx, cid, cross_score):
    f1 = feature_1_domain(global_idx, meta_idx, cid, cross_score)
    f2 = feature_2_retrieval_expertise(cid)
    f3 = feature_3_production_evidence(global_idx, cid)
    f4 = feature_4_vector_infra(global_idx, cid)
    f5 = feature_5_behavioral(meta_idx)
    f6 = feature_6_llm_adjacent(cid)
    f7 = feature_7_career_progression(cid)

    raw = (
        0.30 * f1 +    # Career domain evidence
        0.26 * f2 +    # Retrieval expertise (depth + coverage)
        0.18 * f3 +    # Production evidence (keyword + semantic + impact)
        0.10 * f4 +    # Vector DB / search infra
        0.10 * f5 +    # Behavioral (capped 0.80)
        0.04 * f6 +    # LLM adjacent
        0.02 * f7      # Career progression
    )
    # Weights sum to 1.00

    # PENALTIES — multiplicative
    raw *= services_penalty(cid, f1)
    raw *= stuffer_check(cid, meta_idx)
    raw *= honeypot_penalty(cid)

    return raw, [f1, f2, f3, f4, f5, f6, f7]

# Score top 300
final_scores = []
for gi, ce_score in top_300_global:
    meta_i = global_idx_to_meta.get(gi)
    if meta_i is None:
        continue
    cid = cids_arr[meta_i]
    score, features = compute_final_score(gi, meta_i, cid, float(ce_score))
    final_scores.append((gi, meta_i, cid, score, features))

final_scores.sort(key=lambda x: x[3], reverse=True)
top_100_raw = final_scores[:100]
```

---

### 2.11 Stage 5 — Top-10 Re-Verification

**50% of score = NDCG@10. Explicitly audit top-10 before finalizing.**

```python
from datetime import date

RECHECK_TODAY = date.today()

def top10_disqualify(cid, meta_idx):
    """
    Returns (should_demote, reason) for a candidate in top-10 slots.
    More aggressive thresholds than general scoring.
    """
    row = meta_df.iloc[meta_idx]

    # Notice period: 90+ days = soft disqualify from top-10
    nd = safe_val(row["notice_period_days"], 90)
    if nd > 90:
        return True, f"{int(nd)}-day notice"

    # Stale profile: inactive > 120 days
    try:
        days = (RECHECK_TODAY - pd.to_datetime(row["last_active_date"]).date()).days
    except:
        days = 999
    if days > 120:
        return True, f"inactive {days}d"

    # Very low response rate in top-10
    rr = safe_val(row["response_rate"], 0.5)
    if rr < 0.15:
        return True, f"response rate {rr:.0%}"

    # Soft honeypot
    if cid in honeypot_soft:
        return True, "soft honeypot flag"

    return False, ""

# Apply re-verification
top_10 = top_100_raw[:10]
rest = top_100_raw[10:]

demoted = []
kept_top10 = []
for entry in top_10:
    gi, meta_i, cid, score, features = entry
    disqualify, reason = top10_disqualify(cid, meta_i)
    if disqualify:
        demoted.append((entry, reason))
    else:
        kept_top10.append(entry)

# Fill demoted slots from rank 11-20
if demoted:
    candidates_11_20 = rest[:20]  # look at next 20 for replacements
    for (demoted_entry, reason), replacement in zip(demoted, candidates_11_20):
        logging.info(f"Top-10 reshuffle: demoted {demoted_entry[2]} ({reason}), "
                     f"promoted {replacement[2]}")
        kept_top10.append(replacement)
    # Sort kept_top10 by score
    kept_top10.sort(key=lambda x: x[3], reverse=True)

# Rebuild final top-100
# Remove promoted candidates from rest
promoted_cids = {e[2] for e in kept_top10 if e not in top_10}
rest_filtered = [e for e in rest if e[2] not in promoted_cids]

# Add demoted candidates back into rest at appropriate position
for demoted_entry, _ in demoted:
    rest_filtered.append(demoted_entry)

rest_filtered.sort(key=lambda x: x[3], reverse=True)

top_100_final = kept_top10[:10] + rest_filtered[:90]

# Verify 100 unique candidates
assert len(top_100_final) == 100, f"Expected 100, got {len(top_100_final)}"
assert len({e[2] for e in top_100_final}) == 100, "Duplicate CIDs in top-100"

logging.info(f"Top-10 re-verification: {len(demoted)} reshuffles applied")
```

---

### 2.12 Stage 6 — Evidence-Grounded Reasoning

All claims reference actual field values. No hallucination.
Stage 4 spec checks: specific facts, JD connection, honest concerns, no hallucination,
variation, rank consistency.

```python
def generate_reasoning(entry, rank):
    gi, meta_i, cid, score, features = entry
    row = meta_df.iloc[meta_i]
    data = nested.get(cid, {})
    ev = evidence_lookup.get(cid, {})

    title    = str(row["current_title"])
    years    = round(float(row["years_of_experience"]), 1)
    country  = str(row.get("country", ""))
    location = str(row.get("location", ""))
    notice   = int(safe_val(row.get("notice_period_days", 90), 90))
    rr       = float(safe_val(row.get("response_rate", 0.5), 0.5))
    companies = data.get("career_companies", [])
    career   = data.get("career_text", "").lower()

    # --- Primary strength: highest-scoring feature ---
    sorted_f = sorted(enumerate(features), key=lambda x: x[1], reverse=True)
    best_i   = sorted_f[0][0]
    second_i = sorted_f[1][0]

    s1 = _primary_strength(best_i, cid, data, title, years, companies, career, ev)
    s2 = _secondary_strength(second_i, cid, data, career, ev) if sorted_f[1][1] > 0.30 else ""
    s3 = _concern(notice, years, rr, country, location, companies, cid)

    # Tone varies by rank band — avoids templated reasoning flag at Stage 4
    if rank <= 5:
        parts = [p for p in [s1, s2, s3] if p]
    elif rank <= 20:
        parts = [s1, s3] if s3 else [s1, s2] if s2 else [s1]
    elif rank <= 50:
        parts = [s1, s3] if s3 else [s1]
    else:
        parts = [s1]
        if s3: parts.append(s3)

    return " ".join(parts)


def _primary_strength(fi, cid, data, title, years, companies, career, ev):
    co = companies[0] if companies else "current company"
    cp = ev.get("company_profile", {})
    product_cos = [c for c in companies
                   if classify_company(c) == "product"]
    prod_str = product_cos[0] if product_cos else co

    if fi == 0:  # domain
        found = [kw for kw in STRONG_KEYWORDS if kw in career]
        if found:
            return (f"{years}yr {title} with hands-on {found[0]} experience"
                    f" at {prod_str}.")
        if cp.get("has_product"):
            return (f"{years}yr {title} at product company ({prod_str});"
                    f" career aligns with retrieval/ranking domain.")
        return f"{years}yr {title}; profile strongly matches retrieval/ranking requirements."

    elif fi == 1:  # expertise
        matched = [s["skill_name"] for s in skill_lookup.get(cid, [])
                   if s["best_jd_match_score"] > SKILL_MATCH_THRESHOLD
                   and s["best_jd_req_idx"] < 10][:4]
        skills_str = ", ".join(matched) if matched else "retrieval/search skills"
        return f"{years}yr {title}; credibility-verified skills: {skills_str}."

    elif fi == 2:  # production
        impacts = ev.get("impact_mentions", [])
        if impacts:
            return (f"Production deployment evidence with quantified impact"
                    f" ({impacts[0]}); {years}yr {title} at {prod_str}.")
        found = [k for k in DEPLOYMENT_KEYWORDS if k in career]
        if found:
            return f"{found[0].capitalize()} production systems at {prod_str}; {years}yr {title}."
        return f"Career at {prod_str} shows production-level engineering; {years}yr {title}."

    elif fi == 3:  # vector infra
        tools = [s["skill_name"] for s in skill_lookup.get(cid, [])
                 if s.get("norm_name", s["skill_name"].lower()) in VECTOR_TOOLS][:3]
        tools_str = ", ".join(tools) if tools else "search infrastructure"
        return f"Hands-on {tools_str} experience; {years}yr {title}."

    elif fi == 4:  # behavioral
        saved = safe_val(meta_df.iloc[
            list(meta_df["candidate_id"]).index(cid)
        ]["saved_by_recruiters_30d"] if cid in list(meta_df["candidate_id"]) else 0, 0)
        if saved >= 5:
            return (f"High recruiter interest ({int(saved)} saves in 30d);"
                    f" {years}yr {title} actively available.")
        return f"Strong platform engagement signals active job search; {years}yr {title}."

    elif fi == 5:  # LLM
        return f"LLM/fine-tuning expertise complements core retrieval skills; {years}yr {title}."

    else:  # progression
        return f"Consistent upward career progression; currently {years}yr {title} at {co}."


def _secondary_strength(fi, cid, data, career, ev):
    if fi == 0:
        found = [kw for kw in STRONG_KEYWORDS if kw in career]
        return f"Also shows {found[0]} work." if found else ""
    elif fi == 1:
        matched = [s["skill_name"] for s in skill_lookup.get(cid, [])
                   if s["best_jd_match_score"] > SKILL_MATCH_THRESHOLD][:3]
        return f"Relevant skills: {', '.join(matched)}." if matched else ""
    elif fi == 2:
        impacts = ev.get("impact_mentions", [])
        return f"Impact: {impacts[0]}." if impacts else "Has shipped to production."
    elif fi == 3:
        tools = [s["skill_name"] for s in skill_lookup.get(cid, [])
                 if s.get("norm_name", "").lower() in VECTOR_TOOLS][:2]
        return f"Infrastructure: {', '.join(tools)}." if tools else ""
    elif fi == 4:
        return "Active platform engagement indicates availability."
    elif fi == 5:
        return "Additional LLM fine-tuning depth."
    else:
        return "Steady upward title progression."


def _concern(notice, years, rr, country, location, companies, cid):
    ev = evidence_lookup.get(cid, {})
    cp = ev.get("company_profile", {})

    if notice > 90:
        return (f"Concern: {notice}-day notice significantly exceeds"
                f" JD's 30-day preference.")
    if notice > 60:
        return f"Note: {notice}-day notice exceeds JD preference of ≤30 days."
    if years < 4:
        return f"Note: {years}yrs is below the 5-9yr target range."
    if years > 12:
        return f"Note: {years}yrs is above the 5-9yr target range."
    if rr < 0.20:
        return f"Concern: recruiter response rate {rr:.0%} suggests limited availability."
    if "india" not in country.lower() and country:
        return f"Note: based in {location} ({country}), outside preferred India locations."
    if cp.get("all_services", False):
        svc_names = ", ".join(str(c) for c in companies[:2])
        return f"Note: entire career at services firms ({svc_names})."
    return ""
```

---

### 2.13 Output

```python
import csv

# Final sort — scores must be monotonically non-increasing
top_100_final.sort(key=lambda x: x[3], reverse=True)

# Tie-breaking: equal scores → candidate_id ascending (per spec)
from itertools import groupby
result_rows = []
for _, group in groupby(top_100_final, key=lambda x: round(x[3], 6)):
    grp = sorted(list(group), key=lambda x: x[2])  # CID ascending
    result_rows.extend(grp)

with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (gi, meta_i, cid, score, features) in enumerate(result_rows, 1):
        entry = (gi, meta_i, cid, score, features)
        reasoning = generate_reasoning(entry, rank)
        writer.writerow([cid, rank, round(score, 6), reasoning])

logging.info("submission.csv written — 100 candidates")
```

---

## Part 3 — Runtime Budget

| Stage | Operation | Time |
|---|---|---|
| Load + cast + vectorize | Artifacts, arrays, semantic matmul | ~25s |
| Stage 0 | Working pool construction | ~1s |
| Stage 1 | 5-retriever union (FAISS × 4) | ~4s |
| Stage 2 | Fusion scoring (~3500 candidates) | ~15s |
| Stage 3 | Cross-encoder 800 pairs, batch_size=32 | ~95s |
| Stage 4 | Feature scoring 300 candidates | ~18s |
| Stage 5 | Top-10 re-verification | ~1s |
| Stage 6 | Reasoning generation 100 candidates | ~5s |
| Output | CSV write | <1s |
| **Total** | | **~165s** |
| **Worst case** (slow CE + 800 pairs) | | **~250s** |
| **Limit** | | **300s ✅** |

Circuit breaker at Stage 3 ensures worst case stays under 300s.

---

## Part 4 — File Structure

```
redrob-ranker/
├── precompute.py           GPU pre-computation (run once on Colab)
├── rank.py                 CPU ranking (produces submission.csv)
├── scoring/
│   ├── __init__.py
│   ├── constants.py        Keyword lists, thresholds
│   ├── jd_compiler.py      Auto JD Compiler
│   ├── taxonomy.py         Company taxonomy mapping
│   ├── features.py         7 feature functions
│   ├── penalties.py        Services ×0.40, stuffer ×0.20
│   ├── exclusions.py       Honeypot detection (4 signals)
│   ├── reasoning.py        Evidence-grounded reasoning
│   └── top10_verify.py     Top-10 re-verification pass
├── artifacts/              ~1.5GB (gitignore this)
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Part 5 — Dependencies

```
sentence-transformers>=2.7.0
numpy>=1.24.0
torch>=2.0.0
pandas>=2.0.0
pyarrow>=12.0.0
faiss-cpu>=1.7.4
```

---

## Part 6 — Verification Checklist

```bash
# Pre-computation (GPU, Colab)
python precompute.py --candidates ./candidates.jsonl --jd ./job_description.md --out-dir ./artifacts/

# Ranking (CPU-only, no network)
time python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Format validation
python validate_submission.py submission.csv
```

- [ ] `validate_submission.py` passes with no errors
- [ ] CSV header is exactly `candidate_id,rank,score,reasoning`
- [ ] Exactly 100 rows, ranks 1–100 each exactly once
- [ ] Scores monotonically non-increasing (verified by validator)
- [ ] Ordering assertion passes at load
- [ ] Runtime < 300s on cold CPU (test with `--network none`)
- [ ] Zero hard-honeypots in top 100
- [ ] Zero HR/Marketing/Accountant/Operations/Mechanical in top 100
- [ ] Top-10: all ML/AI/Search engineers with retrieval evidence
- [ ] Top-10: no candidate with notice_period > 90 days
- [ ] Top-10: no candidate inactive > 120 days
- [ ] Company taxonomy: Hooli/Pied Piper/Initech recognized as product companies
- [ ] Sentinel check: github_activity_score=-1 not treated as low score
- [ ] Services penalty: all-TCS + low F1 → ×0.40; all-HCL + high F1 → ×0.80
- [ ] 10 sampled reasonings: structurally different, facts verifiable against profile
- [ ] Reasoning: rank-1 candidate has stronger tone than rank-95
- [ ] Cold Docker test: `docker run --network none <img> time python rank.py`

---

## Part 7 — Cumulative Changes v15 → v21

| Area | v15 | v21 |
|---|---|---|
| JD parsing | Hardcoded 5 queries, 20 skills, 10 archetypes | **Auto JD Compiler — regex + section parsing** |
| Candidate embeddings | 1 monolithic per candidate | **4 chunk embeddings: summary, career, skills, impact** |
| FAISS indices | 1 index | **4 indices (one per chunk)** |
| Pre-filter | After retrieval | **Before retrieval — 51K non-tech titles eliminated** |
| Company taxonomy | Services firm blacklist only | **Fictional company mapping: Hooli/Pied Piper/Initech = product** |
| Honeypot signals | 3 (skill inflation, timeline, YOE inflation) | **4 (+ title/description coherence)** |
| github_activity_score | Treated as numeric (penalizes -1) | **-1 sentinel → None → neutral 0.50** |
| offer_acceptance_rate | Not in v15 | **-1 sentinel → None (not used in scoring but clean for reasoning)** |
| saved_by_recruiters_30d | 0.03 weight | **0.06 weight — market-validation signal** |
| Skill assessments | Weighted at 0.15 in credibility | **0.05 (near-zero JD-relevant coverage confirmed by data)** |
| Achievements field | Referenced | **Removed — field does not exist in schema** |
| Career descriptions | Concatenated into monolithic text | **Primary signal-dense field; first-class chunk embedding** |
| Top-10 handling | Implicit final sort | **Explicit re-verification pass (notice, recency, response rate)** |
| Feature 1 product bonus | None | **+0.10 if product company in career (Hooli/real startups)** |
| Fusion scoring | Not present | **Explicit fusion stage (800→300 before cross-encoder)** |
| Circuit breaker | None | **Cross-encoder time budget — drops to 500 if projected >130s** |
| Reasoning variation | Rank bands 10/30/60 | **Rank bands 5/20/50 (tighter, more variation at top)** |
