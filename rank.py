#!/usr/bin/env python3
"""
rank.py — Redrob AI Candidate Ranking System
Runs on CPU, must complete in <300 seconds, no network calls.

Usage:
  python rank.py --candidates ./candidates.jsonl --artifacts ./artifacts --out ./submission.csv
"""

import argparse

import csv
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd

# Defaults — overridden by CLI args in main()
ARTIFACTS_DIR = "./artifacts"
OUTPUT_PATH = "./submission.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# 1. CONSTANTS
# ============================================================================

JD_CORE = """Senior AI Engineer, founding team. Must have shipped
ranking, search, or recommendation systems to production.
Production experience with embeddings, vector databases, hybrid search.
Python, PyTorch, evaluation frameworks. Product company background.
5-9 years experience. India preferred, hybrid work."""

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

# --- TITLE TIERS ---

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

NON_TECH_TITLES = [
    "hr manager", "human resource", "recruiter",
    "marketing manager", "sales manager", "account manager",
    "operations manager", "accountant", "finance manager",
    "customer support", "copywriter", "graphic designer",
]

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


# ============================================================================
# 2. LOAD ARTIFACTS + SETUP
# ============================================================================

def load_artifacts():
    """Load all precomputed artifacts. Returns dict of globals."""
    art = ARTIFACTS_DIR
    log.info("Loading artifacts...")
    t0 = time.time()

    meta_df = pd.read_parquet(os.path.join(art, "candidates_flat.parquet"))
    nested = pickle.load(open(os.path.join(art, "candidates_nested.pkl"), "rb"))
    cand_embeds_f16 = np.load(os.path.join(art, "candidate_embeddings.npy"))
    archetype_max_raw = np.load(os.path.join(art, "archetype_max_scores.npy"))
    jd_queries = np.load(os.path.join(art, "jd_query_embeddings.npy"))
    jd_skills = np.load(os.path.join(art, "jd_skill_embeddings.npy"))
    skill_lookup = pickle.load(open(os.path.join(art, "skill_matches.pkl"), "rb"))
    honeypot_data = pickle.load(open(os.path.join(art, "honeypot_flags.pkl"), "rb"))

    # Ordering assertion
    saved_order = np.load(os.path.join(art, "candidate_order.npy"), allow_pickle=True)
    assert len(cand_embeds_f16) == len(meta_df), "Embedding/metadata count mismatch"
    assert np.array_equal(saved_order, meta_df["candidate_id"].values), \
        "Embedding/metadata ordering mismatch!"

    # Cast float32 ONCE
    cand_embeds = cand_embeds_f16.astype(np.float32)

    # Map archetype max: cosine [-1,1] → [0,1]
    archetype_max = (archetype_max_raw.astype(np.float32) + 1.0) / 2.0

    # Vectorized semantic: (100K, 5) — all cosine scores at once
    all_semantic_raw = cand_embeds @ jd_queries.T
    all_semantic = (all_semantic_raw + 1.0) / 2.0      # map to [0,1]
    all_semantic_max = all_semantic.max(axis=1)          # best of 5 queries

    # Skill threshold (calibrated during precompute)
    try:
        skill_threshold = float(np.load(os.path.join(art, "skill_threshold.npy"))[0])
    except FileNotFoundError:
        skill_threshold = 0.50
        log.warning("No calibrated threshold found, using default 0.50")

    # FAISS
    use_faiss = False
    faiss_index = None
    try:
        import faiss
        faiss_index = faiss.read_index(os.path.join(art, "candidate.index"))
        use_faiss = True
        log.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")
    except (ImportError, Exception) as e:
        log.warning(f"FAISS not available ({e}), using numpy fallback")

    # O(1) lookups
    cid_to_idx = {cid: i for i, cid in enumerate(meta_df["candidate_id"])}

    # Pre-materialize columns as numpy arrays
    titles_arr = meta_df["current_title"].values
    years_arr = meta_df["years_of_experience"].values
    cids_arr = meta_df["candidate_id"].values

    # Pre-compute TODAY timestamp
    today = pd.Timestamp.now()

    log.info(f"Artifacts loaded in {time.time()-t0:.1f}s")
    log.info(f"Candidates: {len(meta_df)}, Skill threshold: {skill_threshold}")

    return {
        "meta_df": meta_df,
        "nested": nested,
        "cand_embeds": cand_embeds,
        "archetype_max": archetype_max,
        "jd_queries": jd_queries,
        "jd_skills": jd_skills,
        "skill_lookup": skill_lookup,
        "honeypot_hard": honeypot_data["hard"],
        "honeypot_soft": honeypot_data["soft"],
        "all_semantic_raw": all_semantic_raw,
        "all_semantic": all_semantic,
        "all_semantic_max": all_semantic_max,
        "skill_threshold": skill_threshold,
        "use_faiss": use_faiss,
        "faiss_index": faiss_index,
        "cid_to_idx": cid_to_idx,
        "titles_arr": titles_arr,
        "years_arr": years_arr,
        "cids_arr": cids_arr,
        "today": today,
    }


# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def is_non_technical(title):
    """Hard exclude non-technical titles. Returns True if should exclude."""
    t = title.lower()
    is_non_tech = any(kw in t for kw in NON_TECH_TITLES)
    has_tech = any(kw in t for kw in TECH_CARVEOUTS)
    return is_non_tech and not has_tech


def is_honeypot(cid, ctx):
    """O(1) lookup — hard exclude for suspicion >= 3."""
    return cid in ctx["honeypot_hard"]


def runtime_honeypot_check(cid, ctx):
    """
    Runtime honeypot detection — catches profiles missed by precompute.
    Spec: '~80 honeypots with subtly impossible profiles, e.g. expert
    proficiency in 10 skills with 0 years used.'
    Returns: 0.0 (hard exclude), 0.50 (soft penalty), or 1.0 (clean).
    """
    skills = ctx["skill_lookup"].get(cid, [])
    if len(skills) < 5:
        return 1.0  # not enough data to judge

    suspicion = 0

    # Check 1: Advanced/expert proficiency with zero duration
    adv_zero = sum(1 for s in skills
                   if s["proficiency"] in ("advanced", "expert")
                   and s["duration_months"] == 0)
    if adv_zero >= 5:
        suspicion += 2
    elif adv_zero >= 3:
        suspicion += 1

    # Check 2: Many high JD-matching skills with zero endorsements AND zero duration
    ghost_skills = sum(1 for s in skills
                       if s["best_jd_match_score"] > ctx["skill_threshold"]
                       and s["endorsements"] == 0
                       and s["duration_months"] == 0)
    if ghost_skills >= 6:
        suspicion += 2
    elif ghost_skills >= 4:
        suspicion += 1

    # Check 3: Unrealistically broad skill coverage (matches 12+ JD requirements)
    high_match = sum(1 for s in skills if s["best_jd_match_score"] > 0.60)
    if high_match >= 15:
        suspicion += 1

    # Check 4: Many skills overall with uniformly zero duration
    zero_dur = sum(1 for s in skills if s["duration_months"] == 0)
    if zero_dur >= 10 and zero_dur / len(skills) > 0.7:
        suspicion += 1

    if suspicion >= 3:
        return 0.0   # hard exclude
    elif suspicion >= 2:
        return 0.50   # soft penalty
    return 1.0


def honeypot_penalty(cid, ctx):
    """Soft penalty for suspicion == 2. Returns multiplier."""
    if cid in ctx["honeypot_soft"]:
        return 0.50
    return 1.0


def classify_title(title):
    """Returns 1.0 (Tier A), 0.6 (B), 0.3 (C), 0.05 (other)."""
    t = title.lower()
    if any(k in t for k in TIER_A_TITLES):
        return 1.0
    if any(k in t for k in TIER_B_TITLES):
        return 0.6
    if any(k in t for k in TIER_C_TITLES):
        return 0.3
    return 0.05


def score_experience(years):
    """Score experience band. JD sweet spot: 5-9 years."""
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 12.0:
        return 0.75
    if 3.0 <= years < 4.0 or 12.0 < years <= 15.0:
        return 0.5
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


def compute_recency_score_single(last_active, today):
    """Single candidate recency. Returns [0, 1]."""
    try:
        days = (today - pd.to_datetime(last_active)).days
    except Exception:
        return 0.5
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.8
    if days <= 180:
        return 0.5
    if days <= 365:
        return 0.25
    return 0.1


def compute_recency_scores(last_active_series, today):
    """Vectorized recency scoring for entire DataFrame column."""
    dates = pd.to_datetime(last_active_series, errors="coerce")
    days = (today - dates).dt.days.fillna(365)
    scores = pd.Series(0.1, index=days.index)
    scores[days <= 30] = 1.0
    scores[(days > 30) & (days <= 90)] = 0.8
    scores[(days > 90) & (days <= 180)] = 0.5
    scores[(days > 180) & (days <= 365)] = 0.25
    return scores


def compute_quick_behavioral(meta_df, today):
    """Vectorized quick behavioral score for Stage 1 retrieval."""
    recency = compute_recency_scores(meta_df["last_active_date"], today)
    response = meta_df["recruiter_response_rate"].clip(0, 1).fillna(0.5)
    otw = meta_df["open_to_work"].astype(float).fillna(0.5)
    return 0.50 * response + 0.30 * recency + 0.20 * otw


def safe_val(val, default=0.5):
    """Safe value extraction — missing/NaN → default."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def endorse_weight(e):
    """Endorsement count → credibility weight."""
    if e == 0:
        return 0.40
    if e <= 5:
        return 0.70
    if e <= 20:
        return 0.90
    return 1.0


def duration_weight(d):
    """Skill duration (months) → credibility weight."""
    if d == 0:
        return 0.20
    if d <= 6:
        return 0.50
    if d <= 24:
        return 0.80
    return 1.0


def compute_credibility(skill, cid, ctx):
    """
    Weighted average credibility. NOT multiplicative.
    Minimum realistic output: ~0.33. Maximum: 1.0.
    """
    prof = PROF_WEIGHTS.get(skill["proficiency"], 0.60)
    endorse = endorse_weight(skill["endorsements"])
    dur = duration_weight(skill["duration_months"])

    assessments = ctx["nested"].get(cid, {}).get("skill_assessment_scores", {})
    assess_score = assessments.get(skill["skill_name"])
    if assess_score is not None:
        assess = 1.0 if assess_score >= 70 else 0.80 if assess_score >= 40 else 0.50
    else:
        assess = 0.80

    credibility = 0.35 * prof + 0.25 * endorse + 0.25 * dur + 0.15 * assess
    return min(credibility, 1.0)


# --- PENALTIES ---

def services_penalty(cid, f1_score, ctx):
    """
    JD: 'only worked at consulting firms — will not move forward.'
    - All services + low domain (f1 < 0.30): ×0.40
    - All services + some domain (f1 >= 0.30): ×0.80
    """
    companies = ctx["nested"].get(cid, {}).get("career_companies", [])
    non_empty = [co for co in companies if co and co.strip()]
    if not non_empty:
        return 1.0
    all_svc = all(
        any(sf in co.lower() for sf in SERVICES_FIRMS) for co in non_empty
    )
    if all_svc:
        if f1_score < 0.30:
            return 0.40
        return 0.80
    return 1.0


def stuffer_check(cid, idx, ctx):
    """
    ×0.20 for non-tech-adjacent titles with many JD-matching skills
    but low average credibility. Catches keyword stuffers.
    """
    title_tier = classify_title(str(ctx["titles_arr"][idx]))
    if title_tier > 0.3:  # Tier A or B — unlikely stuffer
        return 1.0

    skills = ctx["skill_lookup"].get(cid, [])
    matched = [s for s in skills
               if s["best_jd_match_score"] > ctx["skill_threshold"]]
    if len(matched) < 5:
        return 1.0

    mean_cred = sum(compute_credibility(s, cid, ctx) for s in matched) / len(matched)
    if mean_cred < 0.45:
        return 0.20
    return 1.0


# --- CROSS-ENCODER TEXT BUILDER ---

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


def build_ce_text(cid, idx, ctx):
    """Build candidate text for cross-encoder input (~375 tokens)."""
    data = ctx["nested"].get(cid, {})
    parts = []

    summary = str(ctx["meta_df"].iloc[idx].get("summary", "") or "")
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


# ============================================================================
# 4. STAGE 1: UNION RETRIEVAL + SCORING
# ============================================================================

def stage_1(ctx):
    """Retrieve candidates, hard-exclude, score, return top-N."""
    log.info("=== STAGE 1: Retrieval ===")
    t0 = time.time()

    # --- Semantic pool (FAISS or numpy) ---
    semantic_pool = set()
    if ctx["use_faiss"]:
        import faiss
        for q in ctx["jd_queries"]:
            q32 = q.reshape(1, -1).astype(np.float32)
            _, indices = ctx["faiss_index"].search(q32, 1000)
            semantic_pool.update(indices[0].tolist())
        log.info(f"FAISS semantic pool: {len(semantic_pool)}")
    else:
        for i in range(5):
            sims = ctx["all_semantic_raw"][:, i]
            top_k_idx = np.argpartition(sims, -1000)[-1000:]
            semantic_pool.update(top_k_idx.tolist())
        log.info(f"Numpy semantic pool: {len(semantic_pool)}")

    # --- Skill-match pool ---
    skill_threshold = ctx["skill_threshold"]
    must_have_counts = np.zeros(len(ctx["meta_df"]))
    for i, cid in enumerate(ctx["meta_df"]["candidate_id"]):
        skills = ctx["skill_lookup"].get(cid, [])
        must_have_counts[i] = sum(1 for s in skills
                                   if s["best_jd_req_idx"] < 10
                                   and s["best_jd_match_score"] > skill_threshold)
    skill_top500 = set(np.argpartition(must_have_counts, -500)[-500:].tolist())
    log.info(f"Skill pool: {len(skill_top500)}")

    # --- Behavioral pool ---
    behavioral_arr = compute_quick_behavioral(ctx["meta_df"], ctx["today"]).values
    behav_top200 = set(np.argpartition(behavioral_arr, -200)[-200:].tolist())

    # --- UNION ---
    candidate_pool = semantic_pool | skill_top500 | behav_top200
    log.info(f"Union pool: {len(candidate_pool)}")

    # --- HARD EXCLUSIONS ---
    excluded_honeypots_pre = 0
    excluded_honeypots_rt = 0
    excluded_titles = 0
    soft_penalized = 0
    filtered_pool = []

    for idx in candidate_pool:
        cid = ctx["cids_arr"][idx]
        title = str(ctx["titles_arr"][idx])

        # Precompute honeypot check
        if is_honeypot(cid, ctx):
            excluded_honeypots_pre += 1
            continue

        # Runtime honeypot check (catches what precompute missed)
        hp_mult = runtime_honeypot_check(cid, ctx)
        if hp_mult == 0.0:
            excluded_honeypots_rt += 1
            continue
        elif hp_mult < 1.0:
            soft_penalized += 1

        if is_non_technical(title):
            excluded_titles += 1
            continue

        filtered_pool.append(idx)

    log.info(f"Excluded {excluded_honeypots_pre} precompute honeypots, "
             f"{excluded_honeypots_rt} runtime honeypots, "
             f"{excluded_titles} non-technical titles, "
             f"{soft_penalized} soft-penalized. "
             f"Remaining: {len(filtered_pool)}")

    # --- Stage 1 scoring ---
    def stage1_score(idx):
        cid = ctx["cids_arr"][idx]

        semantic = float(ctx["all_semantic_max"][idx])
        archetype = float(ctx["archetype_max"][idx])

        skills = ctx["skill_lookup"].get(cid, [])
        matched = [s for s in skills
                   if s["best_jd_req_idx"] < 10
                   and s["best_jd_match_score"] > skill_threshold]
        skill = min(len(matched) / 10.0, 1.0)

        career_text = ctx["nested"].get(cid, {}).get("career_text", "").lower()
        pattern = compute_domain_keyword_score(career_text)

        title_score = classify_title(str(ctx["titles_arr"][idx]))
        behav = float(behavioral_arr[idx])
        exp = score_experience(float(ctx["years_arr"][idx]))

        score = (
            0.25 * semantic +
            0.15 * archetype +
            0.20 * skill +
            0.15 * pattern +
            0.10 * title_score +
            0.08 * behav +
            0.07 * exp
        )
        score *= stuffer_check(cid, idx, ctx)
        return score

    pool_scores = [(idx, ctx["cids_arr"][idx], stage1_score(idx))
                   for idx in filtered_pool]
    pool_scores.sort(key=lambda x: x[2], reverse=True)

    top_k = min(500, len(pool_scores))
    top_n = pool_scores[:top_k]

    log.info(f"Stage 1 selected top {top_k} in {time.time()-t0:.1f}s")
    if top_n:
        log.info(f"  Score range: {top_n[0][2]:.4f} → {top_n[-1][2]:.4f}")

    return top_n, behavioral_arr


# ============================================================================
# 5. STAGE 2: CROSS-ENCODER
# ============================================================================

def stage_2(top_n, ctx):
    """Run cross-encoder on top-N candidates. Returns sigmoid scores."""
    log.info(f"=== STAGE 2: Cross-Encoder ({len(top_n)} pairs) ===")
    t0 = time.time()

    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(
        os.path.join(ARTIFACTS_DIR, "models", "ce"),
        device="cpu",
    )
    log.info(f"Cross-encoder loaded in {time.time()-t0:.1f}s")

    # Build pairs
    pairs = [(JD_CORE, build_ce_text(cid, idx, ctx))
             for idx, cid, _ in top_n]

    # Predict
    t1 = time.time()
    raw_logits = ce_model.predict(pairs, batch_size=32)
    log.info(f"Cross-encoder prediction: {time.time()-t1:.1f}s")

    # Sigmoid normalization
    cross_scores = 1.0 / (1.0 + np.exp(-raw_logits))

    log.info(f"Stage 2 complete in {time.time()-t0:.1f}s")
    log.info(f"  Score range: {cross_scores.min():.4f} → {cross_scores.max():.4f}")

    return cross_scores


# ============================================================================
# 6. STAGE 3: 7-FEATURE SCORING
# ============================================================================

def feature_1(idx, cid, cross_sigmoid, ctx):
    """Career Domain Evidence (weight: 0.32)"""
    # Sub-A: Cross-encoder sigmoid (already [0,1])

    # Sub-B: Archetype max (precomputed, already [0,1])
    arch = float(ctx["archetype_max"][idx])

    # Sub-C: 0.25 keyword + 0.75 semantic (domain-specific query 0 only)
    career_text = ctx["nested"].get(cid, {}).get("career_text", "").lower()
    kw = compute_domain_keyword_score(career_text)
    cos_raw = float(ctx["cand_embeds"][idx] @ ctx["jd_queries"][0])
    sem = (cos_raw + 1.0) / 2.0

    domain = 0.25 * kw + 0.75 * sem

    result = 0.40 * cross_sigmoid + 0.30 * arch + 0.30 * domain
    return max(min(result, 1.0), 0.0)


def feature_2(cid, ctx):
    """Retrieval/Search Expertise (weight: 0.26)"""
    skills = ctx["skill_lookup"].get(cid, [])
    threshold = ctx["skill_threshold"]

    # DEDUP by JD requirement index — keep highest credible score
    seen_jd_reqs = {}
    for s in skills:
        if s["best_jd_match_score"] < threshold:
            continue
        req_idx = s["best_jd_req_idx"]
        cred = compute_credibility(s, cid, ctx)
        credible_score = s["best_jd_match_score"] * cred
        if req_idx not in seen_jd_reqs or credible_score > seen_jd_reqs[req_idx]:
            seen_jd_reqs[req_idx] = credible_score

    must_hits = sorted([v for k, v in seen_jd_reqs.items() if k < 10], reverse=True)
    nice_hits = sorted([v for k, v in seen_jd_reqs.items() if k >= 10], reverse=True)

    # DEPTH: top-5 must-have scores, divisor = max(count, 3)
    top_must = must_hits[:5]
    depth = min(sum(top_must) / max(len(top_must), 3), 1.0) if top_must else 0.0

    # COVERAGE: unique JD must-have requirements matched
    coverage = min(len(must_hits) / 6, 1.0)

    # NICE-TO-HAVE: top-3, divisor = max(count, 2)
    top_nice = nice_hits[:3]
    nice_score = min(sum(top_nice) / max(len(top_nice), 2), 1.0) if top_nice else 0.0

    must_component = 0.60 * depth + 0.40 * coverage
    result = 0.70 * must_component + 0.30 * nice_score
    return min(result, 1.0)


def feature_3(cid, idx, ctx):
    """Production Deployment (weight: 0.15)"""
    ct = ctx["nested"].get(cid, {}).get("career_text", "").lower()

    kw_score = 0.0
    for k in DEPLOYMENT_KEYWORDS:
        if k in ct:
            kw_score += 0.15
    for k in SCALE_KEYWORDS:
        if k in ct:
            kw_score += 0.10
    for k in PRODUCT_ENG_KEYWORDS:
        if k in ct:
            kw_score += 0.08
    for k in ANTI_PRODUCTION_KEYWORDS:
        if k in ct:
            kw_score -= 0.15
    kw_score = max(min(kw_score, 1.0), 0.0)

    cos_raw = float(ctx["cand_embeds"][idx] @ ctx["jd_queries"][2])
    sem = (cos_raw + 1.0) / 2.0

    result = 0.40 * kw_score + 0.60 * sem
    return max(min(result, 1.0), 0.0)


def feature_4(cid, idx, ctx):
    """Vector DB & Infrastructure (weight: 0.10)"""
    skills = ctx["skill_lookup"].get(cid, [])
    career_text = ctx["nested"].get(cid, {}).get("career_text", "").lower()
    threshold = ctx["skill_threshold"]

    # TOOL-BASED COMPONENT
    tools_found = {}
    for s in skills:
        name_lower = s.get("norm_name", s["skill_name"].lower())

        is_tool = (name_lower in VECTOR_TOOLS or
                   s["best_jd_req_idx"] in VECTOR_SEARCH_JD_INDICES)
        if not is_tool:
            continue

        cred = compute_credibility(s, cid, ctx)
        in_desc = name_lower in career_text
        weight = 1.0 if in_desc else 0.6
        effective = cred * weight
        if name_lower not in tools_found or effective > tools_found[name_lower]:
            tools_found[name_lower] = effective

    n = len(tools_found)
    avg_cred = sum(tools_found.values()) / max(n, 1)
    base = {0: 0.0, 1: 0.4, 2: 0.7}.get(n, 1.0)
    tool_score = min(base * avg_cred, 1.0)

    # SEMANTIC COMPONENT — JD query 1: "embeddings vector database..."
    cos_raw = float(ctx["cand_embeds"][idx] @ ctx["jd_queries"][1])
    sem_infra = (cos_raw + 1.0) / 2.0

    result = 0.60 * tool_score + 0.40 * sem_infra
    return max(min(result, 1.0), 0.0)


def feature_5(idx, ctx):
    """Availability & Behavioral (weight: 0.10, capped 0.80)"""
    row = ctx["meta_df"].iloc[idx]

    recency = compute_recency_score_single(row["last_active_date"], ctx["today"])

    rr = safe_val(row["recruiter_response_rate"], 0.5)
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


def feature_6(cid, ctx):
    """LLM & Adjacent (weight: 0.04)"""
    skills = ctx["skill_lookup"].get(cid, [])
    threshold = ctx["skill_threshold"]
    VALUED = {10, 11, 12}
    LESS_VALUED = {13, 14}

    score = 0.0
    for s in skills:
        if s["best_jd_match_score"] < threshold:
            continue
        cred = compute_credibility(s, cid, ctx)
        if s["best_jd_req_idx"] in VALUED:
            score += 0.15 * cred
        elif s["best_jd_req_idx"] in LESS_VALUED:
            score += 0.05 * cred
    return min(score, 1.0)


def get_title_level(title):
    t = title.lower()
    for kw in sorted(TITLE_LEVELS.keys(), key=len, reverse=True):
        if kw in t:
            return TITLE_LEVELS[kw]
    return 2


def feature_7(cid, ctx):
    """Career Progression (weight: 0.03)"""
    career = ctx["nested"].get(cid, {}).get("career_history", [])
    if len(career) < 2:
        return 0.5  # neutral

    career_sorted = sorted(career, key=lambda r: r.get("start_date", ""))
    levels = [get_title_level(r.get("title", "")) for r in career_sorted]

    ups = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i - 1])
    downs = sum(1 for i in range(1, len(levels)) if levels[i] < levels[i - 1])
    total = len(levels) - 1

    if ups > downs:
        return min(0.5 + (ups / total) * 0.5, 1.0)
    elif ups == downs:
        return 0.5
    else:
        return max(0.5 - (downs / total) * 0.3, 0.2)


def experience_penalty(years):
    """Multiplicative penalty for experience far outside 5-9yr sweet spot.
    Stage 1 only uses 0.07 weight which is too weak — 16yr candidates
    still appear in top 30. This adds a meaningful multiplier."""
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 12.0:
        return 0.95
    if 3.0 <= years < 4.0 or 12.0 < years <= 15.0:
        return 0.85
    return 0.75  # <3 or 16+ years


def compute_final_score(idx, cid, cross_sigmoid, ctx):
    """Compute final weighted score with penalties."""
    f1 = feature_1(idx, cid, cross_sigmoid, ctx)
    f2 = feature_2(cid, ctx)
    f3 = feature_3(cid, idx, ctx)
    f4 = feature_4(cid, idx, ctx)
    f5 = feature_5(idx, ctx)
    f6 = feature_6(cid, ctx)
    f7 = feature_7(cid, ctx)

    raw = (
        0.32 * f1 +    # career domain evidence
        0.26 * f2 +    # retrieval expertise
        0.15 * f3 +    # production deployment
        0.10 * f4 +    # vector DB
        0.10 * f5 +    # availability (capped 0.80)
        0.04 * f6 +    # LLM adjacent
        0.03 * f7      # career progression
    )
    # Weights sum to 1.00

    # MULTIPLICATIVE PENALTIES
    raw *= services_penalty(cid, f1, ctx)
    raw *= stuffer_check(cid, idx, ctx)
    raw *= honeypot_penalty(cid, ctx)
    raw *= runtime_honeypot_check(cid, ctx)  # catches precompute misses
    raw *= experience_penalty(float(ctx["years_arr"][idx]))

    return raw, [f1, f2, f3, f4, f5, f6, f7]


def stage_3(top_n, cross_scores, ctx):
    """Score top-N with 7 features, return top 100."""
    log.info(f"=== STAGE 3: Feature Scoring ({len(top_n)} candidates) ===")
    t0 = time.time()

    final_scores = []
    for rank_in_n, (idx, cid, s1) in enumerate(top_n):
        score, features = compute_final_score(idx, cid,
                                               cross_scores[rank_in_n], ctx)
        final_scores.append((idx, cid, score, features))

    # Sort by score descending, then by candidate_id ascending (tie-break)
    final_scores.sort(key=lambda x: (-x[2], x[1]))
    top_100 = final_scores[:100]

    log.info(f"Stage 3 complete in {time.time()-t0:.1f}s")
    if top_100:
        log.info(f"  Score range: {top_100[0][2]:.6f} → {top_100[-1][2]:.6f}")

    return top_100


# ============================================================================
# 7. REASONING GENERATION
# ============================================================================

def _build_skill_detail(skill, cid, ctx):
    """Build a unique detail string for one skill, e.g. 'Elasticsearch (47 endorsements, 44mo)'."""
    name = skill["skill_name"]
    parts = []
    e = skill["endorsements"]
    d = skill["duration_months"]
    if e > 0:
        parts.append(f"{e} endorsements")
    if d > 0:
        parts.append(f"{d}mo")
    # Assessment score if available
    assessments = ctx["nested"].get(cid, {}).get("skill_assessment_scores", {})
    asc = assessments.get(name)
    if asc is not None and asc > 0:
        parts.append(f"{int(asc)}% assessment")
    if parts:
        return f"{name} ({', '.join(parts)})"
    return name


def _get_top_skills_detailed(cid, ctx, req_filter=None, n=3):
    """Get top-N credible skills with detail strings. Always unique per candidate."""
    skills = ctx["skill_lookup"].get(cid, [])
    scored = []
    for s in skills:
        if s["best_jd_match_score"] < ctx["skill_threshold"]:
            continue
        if req_filter is not None and s["best_jd_req_idx"] not in req_filter:
            continue
        cred = compute_credibility(s, cid, ctx)
        scored.append((cred * s["best_jd_match_score"], s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_build_skill_detail(s, cid, ctx) for _, s in scored[:n]]


def _get_domain_keywords_found(career_text):
    """Return which domain keywords appear in career text."""
    found = []
    for kw in STRONG_KEYWORDS:
        if kw in career_text:
            found.append(kw)
            if len(found) >= 2:
                break
    return found


def _strength_sentence(feat_idx, cid, data, title, years,
                       companies, top_skills, career_text, ctx):
    """Primary strength — built from structured, candidate-specific data.
    Every candidate gets a unique sentence because it uses their actual
    skill endorsement counts, durations, assessment scores, and company."""
    co = companies[0] if companies else "their company"

    if feat_idx == 0:  # career domain
        skill_details = _get_top_skills_detailed(cid, ctx, req_filter=set(range(10)), n=2)
        domain_kws = _get_domain_keywords_found(career_text)
        if skill_details and domain_kws:
            return (f"{title} with {years}yrs at {co}; "
                    f"{domain_kws[0]} experience; top skills {' and '.join(skill_details)}.")
        elif skill_details:
            return f"{title} with {years}yrs at {co}; top skills {' and '.join(skill_details)}."
        elif domain_kws:
            return f"{title} with {years}yrs at {co}; {domain_kws[0]} domain experience."
        return f"{title} with {years}yrs at {co}; career aligns with retrieval/ranking requirements."

    elif feat_idx == 1:  # retrieval skills
        skill_details = _get_top_skills_detailed(cid, ctx, req_filter=set(range(10)), n=3)
        if skill_details:
            return f"{title} with {years}yrs; verified skills: {', '.join(skill_details)}."
        return f"{title} with {years}yrs at {co}; skills match retrieval/ranking stack."

    elif feat_idx == 2:  # production
        deploy_kws = [kw for kw in DEPLOYMENT_KEYWORDS if kw in career_text][:1]
        scale_kws = [kw for kw in SCALE_KEYWORDS if kw in career_text][:1]
        evidence = deploy_kws + scale_kws
        if evidence:
            return f"{title} at {co} with {years}yrs; career shows {' and '.join(evidence)}."
        return f"{title} at {co} with {years}yrs; production engineering background."

    elif feat_idx == 3:  # vector DB
        tools = _get_top_skills_detailed(cid, ctx, n=3)
        tool_names = [s["skill_name"] for s in ctx["skill_lookup"].get(cid, [])
                      if s.get("norm_name", s["skill_name"].lower()) in VECTOR_TOOLS][:2]
        if tool_names:
            return f"{title} with {years}yrs at {co}; hands-on with {', '.join(tool_names)}."
        return f"{title} with {years}yrs; vector search infrastructure experience."

    elif feat_idx == 4:  # behavioral
        row = ctx["meta_df"].iloc[ctx["cid_to_idx"][cid]]
        rr = safe_val(row.get("recruiter_response_rate", 0.5), 0.5)
        gh = safe_val(row.get("github_activity_score", -1), -1)
        nd = int(safe_val(row.get("notice_period_days", 90), 90))
        parts = [f"{title} with {years}yrs at {co}"]
        if rr >= 0.7:
            parts.append(f"response rate {rr:.0%}")
        if gh > 50:
            parts.append(f"GitHub activity {int(gh)}")
        if nd <= 30:
            parts.append(f"{nd}-day notice")
        return "; ".join(parts) + "."

    elif feat_idx == 5:  # LLM
        skill_details = _get_top_skills_detailed(cid, ctx, req_filter={10, 11, 12, 13, 14}, n=2)
        if skill_details:
            return f"{title} with {years}yrs; LLM-adjacent: {', '.join(skill_details)}."
        return f"{title} with {years}yrs; LLM/adjacent ML expertise complements core skills."

    else:  # progression
        return f"Strong career progression to {title} with {years}yrs at {co}."


def _secondary_sentence(feat_idx, cid, data, top_skills, career_text, ctx):
    """Supporting evidence — shorter, complementary to primary. Uses structured data."""
    if feat_idx == 0:
        domain_kws = _get_domain_keywords_found(career_text)
        if domain_kws:
            return f"Career includes {domain_kws[0]} work."
        return ""

    elif feat_idx == 1:
        skill_details = _get_top_skills_detailed(cid, ctx, n=2)
        return f"Relevant: {', '.join(skill_details)}." if skill_details else ""

    elif feat_idx == 2:
        deploy_kws = [kw for kw in DEPLOYMENT_KEYWORDS if kw in career_text][:1]
        return f"Evidence of {deploy_kws[0]}." if deploy_kws else ""

    elif feat_idx == 3:
        tools = [s["skill_name"] for s in ctx["skill_lookup"].get(cid, [])
                 if s.get("norm_name", s["skill_name"].lower()) in VECTOR_TOOLS][:2]
        return f"Hands-on with {', '.join(tools)}." if tools else ""

    elif feat_idx == 4:
        return "Strong platform engagement signals."

    elif feat_idx == 5:
        return "Additional LLM/NLP experience."

    else:
        return "Consistent career growth."


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

    non_empty = [co for co in companies if co and co.strip()]
    if non_empty:
        all_svc = all(any(sf in co.lower() for sf in SERVICES_FIRMS) for co in non_empty)
        if all_svc:
            return f"Note: entire career at services firms ({', '.join(str(c) for c in non_empty[:2])})."

    return ""


def generate_reasoning(cid, rank, features, idx, ctx):
    """Build data-driven reasoning string."""
    data = ctx["nested"].get(cid, {})
    row = ctx["meta_df"].iloc[idx]

    title = str(row["current_title"])
    years = round(float(row["years_of_experience"]), 1)
    country = str(row.get("country", "") or "")
    location_str = str(row.get("location", "") or "")
    location = f"{location_str}, {country}" if country else location_str
    notice = int(safe_val(row.get("notice_period_days", 90), 90))
    response_rate = float(safe_val(row.get("recruiter_response_rate", 0.5), 0.5))
    companies = data.get("career_companies", [])[:3]
    top_skills = data.get("skill_names", [])[:5]
    career_text = data.get("career_text", "").lower()

    # SENTENCE 1: Primary strength (highest WEIGHTED contribution, not raw value)
    FEATURE_WEIGHTS = [0.32, 0.26, 0.15, 0.10, 0.10, 0.04, 0.03]
    sorted_f = sorted(enumerate(features),
                      key=lambda x: x[1] * FEATURE_WEIGHTS[x[0]], reverse=True)
    best_i, best_v = sorted_f[0]
    s1 = _strength_sentence(best_i, cid, data, title, years,
                            companies, top_skills, career_text, ctx)

    # SENTENCE 2: Secondary strength (if significant)
    second_i, second_v = sorted_f[1]
    s2 = ""
    if second_v > 0.3:
        s2 = _secondary_sentence(second_i, cid, data, top_skills, career_text, ctx)

    # SENTENCE 3: Concern (data-driven)
    s3 = _concern_sentence(notice, years, response_rate,
                           country, location, companies)

    # Assemble by rank band
    if rank <= 10:
        parts = [s for s in [s1, s2, s3] if s]
    elif rank <= 30:
        parts = [s1, s3] if s3 else [s1, s2] if s2 else [s1]
    elif rank <= 60:
        parts = [s1, s3] if s3 else [s1]
    else:
        parts = [s1]
        if s3:
            parts.append(s3)

    # Clean: remove double spaces, ensure no newlines (CSV safety)
    reasoning = " ".join(parts).replace("\n", " ").replace("  ", " ").strip()
    # Truncate to safe length for CSV
    return reasoning[:500]


# ============================================================================
# 8. CSV OUTPUT
# ============================================================================

def write_csv(top_100, ctx):
    """Write submission CSV. Validates format before writing."""
    log.info("=== Writing CSV ===")

    # Re-sort by ROUNDED score to handle tie-breaking after rounding
    # Validator requires: equal scores → candidate_id ascending
    top_100_rounded = [(idx, cid, round(score, 6), features)
                       for idx, cid, score, features in top_100]
    top_100_rounded.sort(key=lambda x: (-x[2], x[1]))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CRITICAL: column order must be candidate_id, rank, score, reasoning
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (idx, cid, score, features) in enumerate(top_100_rounded, 1):
            reasoning = generate_reasoning(cid, rank, features, idx, ctx)
            writer.writerow([cid, rank, score, reasoning])

    log.info(f"CSV written to {OUTPUT_PATH} with {len(top_100)} candidates")

    # Quick self-validation
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == ["candidate_id", "rank", "score", "reasoning"], \
        f"Header mismatch: {header}"
    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    # Check scores are non-increasing
    scores = [float(r[2]) for r in rows]
    cids = [r[0] for r in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], \
            f"Score not non-increasing at rank {i+1}: {scores[i]} < {scores[i+1]}"
        # Check tie-break: equal scores → candidate_id ascending
        if scores[i] == scores[i + 1]:
            assert cids[i] < cids[i + 1], \
                f"Tie-break violated at rank {i+1}: {cids[i]} > {cids[i+1]}"

    log.info("✅ Self-validation passed: header, 100 rows, non-increasing scores, tie-breaks")


# ============================================================================
# 9. MAIN
# ============================================================================

def main():
    global ARTIFACTS_DIR, OUTPUT_PATH

    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System v15"
    )
    parser.add_argument(
        "--candidates", type=str, default=None,
        help="Path to candidates.jsonl (not used directly — artifacts must "
             "already be precomputed. Kept for spec compliance.)"
    )
    parser.add_argument(
        "--artifacts", type=str, default="./artifacts",
        help="Path to precomputed artifacts directory (default: ./artifacts)"
    )
    parser.add_argument(
        "--out", type=str, default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)"
    )
    args = parser.parse_args()

    ARTIFACTS_DIR = args.artifacts
    OUTPUT_PATH = args.out

    total_t0 = time.time()
    log.info("=" * 60)
    log.info("RANK.PY — Redrob AI Candidate Ranking System v15")
    log.info("=" * 60)
    log.info(f"Artifacts: {ARTIFACTS_DIR}")
    log.info(f"Output:    {OUTPUT_PATH}")

    # Load artifacts
    ctx = load_artifacts()

    # Stage 1: Retrieval + filtering + scoring → top 500
    top_n, behavioral_arr = stage_1(ctx)

    # Stage 2: Cross-encoder on top 500
    cross_scores = stage_2(top_n, ctx)

    # Stage 3: 7-feature scoring → top 100
    top_100 = stage_3(top_n, cross_scores, ctx)

    # Write CSV with reasoning
    write_csv(top_100, ctx)

    # Summary
    total_time = time.time() - total_t0
    log.info(f"\n{'='*60}")
    log.info(f"RANKING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    log.info(f"{'='*60}")

    if total_time > 300:
        log.warning(f"\u26a0\ufe0f OVER BUDGET: {total_time:.0f}s > 300s limit!")
    else:
        log.info(f"\u2705 Within budget: {total_time:.0f}s / 300s")

    # Print top 10
    log.info("\n--- TOP 10 ---")
    for rank, (idx, cid, score, features) in enumerate(top_100[:10], 1):
        title = str(ctx["titles_arr"][idx])
        years = float(ctx["years_arr"][idx])
        log.info(f"  #{rank:2d} | {cid} | {score:.6f} | {title} | {years:.1f}yrs")


if __name__ == "__main__":
    main()
