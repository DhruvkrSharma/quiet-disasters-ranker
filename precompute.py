#!/usr/bin/env python3
"""
precompute.py — Redrob AI Candidate Ranking System 
Run on Kaggle/Colab (GPU) to generate all artifacts for offline ranking.
"""

# ============================================================================
# CONFIGURE THESE PATHS FOR YOUR ENVIRONMENT
# ============================================================================
CANDIDATES_PATH = "/kaggle/input/datasets/abhinavshakya2005/candidates-jsonl/candidates.jsonl"
OUT_DIR = "/kaggle/working/artifacts/"
# ============================================================================

import json
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# 1. JD CONSTANTS (hardcoded for this specific JD — Phase 2 will derive these)
# ============================================================================

JD_CORE = (
    "Senior AI Engineer for talent intelligence platform. "
    "Own ranking, retrieval, and matching systems. "
    "Production experience with embeddings-based retrieval, vector databases, "
    "hybrid search, evaluation frameworks (NDCG, MRR, MAP). "
    "Ship v2 ranking system, set up offline benchmarks and A/B testing. "
    "5-9 years experience, product company preferred, strong Python."
)

JD_QUERIES = [
    # Query 0: Core domain — search/ranking/retrieval
    "search ranking recommendation retrieval information retrieval candidate matching "
    "talent discovery relevance scoring learning to rank",
    # Query 1: Embeddings & vector DB infrastructure
    "embeddings vector database FAISS Pinecone Milvus Weaviate Qdrant OpenSearch "
    "Elasticsearch hybrid search dense retrieval ANN approximate nearest neighbor",
    # Query 2: Production deployment & shipping
    "shipped production deployed real users scale serving latency throughput "
    "inference optimization model serving API endpoint",
    # Query 3: ML/AI core skills
    "machine learning deep learning PyTorch TensorFlow transformers NLP "
    "natural language processing fine-tuning BERT sentence-transformers",
    # Query 4: LLM & modern AI
    "large language models LLM GPT fine-tuning LoRA PEFT RAG "
    "retrieval augmented generation prompt engineering agentic AI",
]

JD_SKILLS = [
    # Must-have (indices 0-9)
    "embeddings-based retrieval systems",
    "vector databases and ANN search",
    "hybrid search infrastructure",
    "search ranking and relevance",
    "evaluation frameworks NDCG MRR MAP",
    "Python programming",
    "PyTorch TensorFlow Keras scikit-learn deep learning ML frameworks",
    "recommendation systems",
    "information retrieval",
    "sentence-transformers and embedding models",
    # Nice-to-have (indices 10-19)
    "LLM fine-tuning LoRA QLoRA PEFT",
    "learning-to-rank XGBoost neural",
    "HR-tech recruiting marketplace",
    "distributed systems inference optimization",
    "open-source AI ML contributions",
    "FAISS vector indexing",
    "Elasticsearch OpenSearch",
    "Pinecone Weaviate Qdrant Milvus",
    "A/B testing experimentation",
    "data pipeline Spark Airflow",
]

ARCHETYPES = [
    "Search engineer who built and shipped production search and ranking systems "
    "at a product company, handling millions of queries",
    "ML engineer specializing in embeddings, retrieval, and recommendation systems "
    "with hands-on vector database experience",
    "AI engineer who deployed production ranking models with evaluation frameworks "
    "like NDCG and MRR at scale",
    "Senior software engineer with deep NLP experience building information "
    "retrieval and candidate matching systems",
    "Full-stack ML engineer who built end-to-end search pipelines from data "
    "ingestion through serving with hybrid retrieval",
    "Applied scientist who transitioned to engineering, shipping retrieval and "
    "ranking systems to real users",
    "Platform engineer who built ML infrastructure for embedding generation, "
    "vector indexing, and model serving",
    "Recommendation system engineer experienced with collaborative filtering, "
    "content-based retrieval, and hybrid approaches",
    "NLP engineer specializing in semantic search, query understanding, and "
    "relevance tuning for production systems",
    "Tech lead who architected and shipped search/ranking/matching systems "
    "at an AI-first product company",
]






# ============================================================================
# 3. LOAD CANDIDATES
# ============================================================================

def load_candidates(path):
    """Load candidates from JSONL. Returns list in file order."""
    log.info(f"Loading candidates from {path}...")
    t0 = time.time()
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    log.info(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")
    return candidates


# ============================================================================
# 4. SAVE MODELS LOCALLY (for offline ranking)
# ============================================================================

def save_models(out_dir):
    """Download and save bi-encoder + cross-encoder models."""
    from sentence_transformers import SentenceTransformer, CrossEncoder

    bi_path = os.path.join(out_dir, "models", "bi")
    ce_path = os.path.join(out_dir, "models", "ce")

    if not os.path.exists(os.path.join(bi_path, "config.json")):
        log.info("Downloading bi-encoder: all-MiniLM-L6-v2 ...")
        bi_model = SentenceTransformer("all-MiniLM-L6-v2")
        bi_model.save(bi_path)
        log.info(f"Bi-encoder saved to {bi_path}")
    else:
        log.info(f"Bi-encoder already exists at {bi_path}")
        bi_model = SentenceTransformer(bi_path)

    if not os.path.exists(os.path.join(ce_path, "config.json")):
        log.info("Downloading cross-encoder: ms-marco-MiniLM-L-6-v2 ...")
        ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ce_model.save(ce_path)
        log.info(f"Cross-encoder saved to {ce_path}")
    else:
        log.info(f"Cross-encoder already exists at {ce_path}")

    return bi_model


# ============================================================================
# 5. ENCODE JD ARTIFACTS
# ============================================================================

def encode_jd_artifacts(bi_model, out_dir):
    """Encode JD queries, skills, and archetypes."""
    log.info("Encoding JD queries...")
    jd_query_embs = bi_model.encode(JD_QUERIES, normalize_embeddings=True)
    np.save(os.path.join(out_dir, "jd_query_embeddings.npy"), jd_query_embs)

    log.info("Encoding JD skills...")
    jd_skill_embs = bi_model.encode(JD_SKILLS, normalize_embeddings=True)
    np.save(os.path.join(out_dir, "jd_skill_embeddings.npy"), jd_skill_embs)

    log.info("Encoding archetypes...")
    arch_embs = bi_model.encode(ARCHETYPES, normalize_embeddings=True)
    np.save(os.path.join(out_dir, "archetype_embeddings.npy"), arch_embs)

    log.info("JD artifacts saved.")
    return jd_query_embs, jd_skill_embs, arch_embs


# ============================================================================
# 6. BUILD CANDIDATE EMBEDDING TEXT
# ============================================================================

def build_embedding_text(candidate):
    """Build labeled-section text for embedding. ~500 tokens."""
    p = candidate.get("profile", {})
    parts = []

    # Skills section (most important for semantic matching)
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Experience section
    career = candidate.get("career_history", [])
    exp_parts = []
    for role in career[:3]:  # top 3 roles
        title = role.get("title", "")
        company = role.get("company", "")
        desc = (role.get("description", "") or "")[:250]
        if title:
            exp_parts.append(f"{title} at {company}: {desc}")
    if exp_parts:
        parts.append("Experience: " + " | ".join(exp_parts))

    # Summary section
    summary = p.get("summary", "") or ""
    if summary:
        parts.append("Summary: " + summary[:300])

    text = " ".join(parts)
    return text[:2000]  # cap total length


# ============================================================================
# 7. ENCODE CANDIDATES
# ============================================================================

def encode_candidates(bi_model, candidates, out_dir):
    """Encode all candidates, compute archetype max scores."""
    log.info("Building candidate embedding texts...")
    texts = [build_embedding_text(c) for c in candidates]

    # Deduplicate texts for faster encoding
    unique_texts = list(set(texts))
    log.info(f"Unique texts: {len(unique_texts)} / {len(texts)}")

    log.info("Encoding candidates (this may take a while)...")
    t0 = time.time()
    unique_embs = bi_model.encode(
        unique_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    )
    log.info(f"Encoding done in {time.time()-t0:.1f}s")

    # Map back to candidate order
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    candidate_embs = np.array([unique_embs[text_to_idx[t]] for t in texts])

    # Save as float16 to reduce size
    candidate_embs_f16 = candidate_embs.astype(np.float16)
    np.save(os.path.join(out_dir, "candidate_embeddings.npy"), candidate_embs_f16)
    log.info(f"Candidate embeddings: {candidate_embs_f16.shape}, "
             f"{candidate_embs_f16.nbytes/1024/1024:.1f}MB")

    # Compute archetype max scores
    log.info("Computing archetype max scores...")
    arch_embs = np.load(os.path.join(out_dir, "archetype_embeddings.npy"))
    # candidate_embs is still float32 here
    arch_scores = candidate_embs @ arch_embs.T  # (N, 10)
    archetype_max = arch_scores.max(axis=1)  # (N,)
    np.save(os.path.join(out_dir, "archetype_max_scores.npy"),
            archetype_max.astype(np.float16))
    log.info(f"Archetype max scores: shape={archetype_max.shape}")

    # Save candidate order
    cids = np.array([c["candidate_id"] for c in candidates])
    np.save(os.path.join(out_dir, "candidate_order.npy"), cids)
    log.info("Candidate order saved.")

    return candidate_embs


# ============================================================================
# 8. BUILD FAISS INDEX
# ============================================================================

def build_faiss_index(candidate_embs, out_dir):
    """Build FAISS index for fast retrieval."""
    try:
        import faiss
        log.info("Building FAISS index...")
        t0 = time.time()
        dim = candidate_embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized
        index.add(candidate_embs.astype(np.float32))
        faiss.write_index(index, os.path.join(out_dir, "candidate.index"))
        log.info(f"FAISS index built in {time.time()-t0:.1f}s, "
                 f"{index.ntotal} vectors")
    except ImportError:
        log.warning("FAISS not installed. Numpy fallback will be used at rank time.")


# ============================================================================
# 9. SKILL MATCHING
# ============================================================================

def compute_skill_matches(bi_model, candidates, jd_skill_embs, out_dir):
    """Match each candidate's skills to JD requirements."""
    log.info("Computing skill matches...")
    t0 = time.time()

    # Collect all unique skill names across all candidates
    all_skill_names = set()
    for c in candidates:
        for s in c.get("skills", []):
            name = s.get("name", "").strip()
            if name:
                all_skill_names.add(name.lower())

    unique_skills = sorted(all_skill_names)
    log.info(f"Unique skills across all candidates: {len(unique_skills)}")

    # Encode all unique skills
    skill_embs = bi_model.encode(unique_skills, normalize_embeddings=True,
                                  batch_size=512, show_progress_bar=True)
    skill_to_emb = dict(zip(unique_skills, skill_embs))

    # Match each candidate's skills to JD
    skill_lookup = {}
    for c in candidates:
        cid = c["candidate_id"]
        matches = []
        for s in c.get("skills", []):
            name = s.get("name", "").strip()
            if not name:
                continue
            norm = name.lower()
            emb = skill_to_emb.get(norm)
            if emb is None:
                continue

            # Cosine similarity to each JD skill
            sims = emb @ jd_skill_embs.T
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            matches.append({
                "skill_name": name,
                "norm_name": norm,
                "best_jd_match_score": best_score,
                "best_jd_req_idx": best_idx,
                "proficiency": s.get("proficiency", "intermediate"),
                "endorsements": s.get("endorsements", 0),
                "duration_months": s.get("duration_months", 0),
            })
        skill_lookup[cid] = matches

    pickle.dump(skill_lookup, open(os.path.join(out_dir, "skill_matches.pkl"), "wb"))
    log.info(f"Skill matches computed in {time.time()-t0:.1f}s")

    return skill_lookup


# ============================================================================
# 10. SKILL THRESHOLD CALIBRATION
# ============================================================================

def calibrate_skill_threshold(bi_model, jd_skill_embs, out_dir):
    """Calibrate the skill match threshold."""
    test_skills = [
        "pytorch", "faiss", "elasticsearch", "embeddings",
        "recommendation systems", "python", "docker", "kubernetes",
        "natural language processing", "transformers",
    ]
    test_embs = bi_model.encode(test_skills, normalize_embeddings=True)

    print("\n=== SKILL THRESHOLD CALIBRATION ===")
    min_relevant_score = 1.0
    for name, emb in zip(test_skills, test_embs):
        sims = emb @ jd_skill_embs.T
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        status = "PASS" if best_score > 0.50 else "BELOW THRESHOLD"
        print(f"  {name:30s} -> JD[{best_idx:2d}] = {best_score:.3f}  ({status})")
        if best_score < min_relevant_score and name in [
            "pytorch", "faiss", "elasticsearch", "embeddings",
            "recommendation systems", "transformers"
        ]:
            min_relevant_score = best_score

    # If any obviously relevant skill is below 0.50, lower threshold
    threshold = 0.50
    if min_relevant_score < 0.50:
        threshold = max(min_relevant_score - 0.02, 0.40)
        print(f"\n  Auto-lowered threshold to {threshold:.2f} "
              f"(min relevant score: {min_relevant_score:.3f})")

    # Re-print with FINAL threshold for clarity
    print(f"\n  === FINAL STATUS (threshold = {threshold:.2f}) ===")
    for name, emb in zip(test_skills, test_embs):
        sims = emb @ jd_skill_embs.T
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        status = "✅ PASS" if best_score >= threshold else "❌ EXCLUDED"
        print(f"  {name:30s} -> JD[{best_idx:2d}] = {best_score:.3f}  ({status})")

    np.save(os.path.join(out_dir, "skill_threshold.npy"), np.array([threshold]))
    print(f"\n  Saved threshold: {threshold}")
    return threshold


# ============================================================================
# 11. METADATA EXTRACTION
# ============================================================================

def extract_metadata(candidates, out_dir):
    """Extract flat metadata to parquet."""
    log.info("Extracting metadata to parquet...")
    t0 = time.time()

    flat = []
    for c in candidates:
        p = c.get("profile", {})
        s = c.get("redrob_signals", {})
        flat.append({
            "candidate_id": c["candidate_id"],
            "current_title": p.get("current_title", ""),
            "headline": p.get("headline", ""),
            "summary": p.get("summary", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "location": p.get("location", ""),
            "country": p.get("country", ""),
            "current_company": p.get("current_company", ""),
            "current_industry": p.get("current_industry", ""),
            "current_company_size": p.get("current_company_size", ""),
            # Redrob signals
            "profile_completeness_score": s.get("profile_completeness_score", 50),
            "signup_date": s.get("signup_date", ""),
            "last_active_date": s.get("last_active_date", ""),
            "open_to_work": s.get("open_to_work_flag", False),
            "recruiter_response_rate": s.get("recruiter_response_rate", 0.5),
            "avg_response_time_hours": s.get("avg_response_time_hours", 72),
            "interview_completion_rate": s.get("interview_completion_rate", 0.5),
            "github_activity_score": s.get("github_activity_score", -1),
            "verified_email": s.get("verified_email", False),
            "verified_phone": s.get("verified_phone", False),
            "linkedin_connected": s.get("linkedin_connected", False),
            "notice_period_days": s.get("notice_period_days", 90),
            "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d", 0),
            "profile_views_received_30d": s.get("profile_views_received_30d", 0),
            "applications_submitted_30d": s.get("applications_submitted_30d", 0),
            "connection_count": s.get("connection_count", 0),
        })

    meta_df = pd.DataFrame(flat)
    meta_df.to_parquet(os.path.join(out_dir, "candidates_flat.parquet"), index=False)
    log.info(f"Parquet saved: {len(meta_df)} rows in {time.time()-t0:.1f}s")
    return meta_df


# ============================================================================
# 12. NESTED DATA EXTRACTION
# ============================================================================

def extract_nested(candidates, out_dir):
    """Extract nested data (career, skills, education) to pickle."""
    log.info("Extracting nested data...")
    t0 = time.time()

    nested_data = {}
    for c in candidates:
        cid = c["candidate_id"]
        career = c.get("career_history", [])
        nested_data[cid] = {
            "career_history": career,
            "career_text": " ".join(
                (r.get("description", "") or "") for r in career
            ),
            "career_companies": [r.get("company", "") for r in career],
            "education": c.get("education", []),
            "skill_names": [s.get("name", "") for s in c.get("skills", [])],
            "skill_assessment_scores": c.get("redrob_signals", {}).get(
                "skill_assessment_scores", {}
            ),
        }

    pickle.dump(nested_data, open(os.path.join(out_dir, "candidates_nested.pkl"), "wb"))
    log.info(f"Nested data saved in {time.time()-t0:.1f}s")
    return nested_data


# ============================================================================
# 13. HONEYPOT PRE-DETECTION
# ============================================================================

def precompute_honeypot_flags(candidates, skill_lookup, out_dir):
    """
    Tiered honeypot detection:
    - 'hard': suspicion >= 3 → hard exclude (clearly fraudulent)
    - 'soft': suspicion == 2 → ×0.50 penalty (possibly messy data)
    """
    log.info("Running honeypot detection...")
    hard_exclude = set()
    soft_penalize = set()

    for c in candidates:
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

    honeypot_data = {"hard": hard_exclude, "soft": soft_penalize}
    pickle.dump(honeypot_data, open(os.path.join(out_dir, "honeypot_flags.pkl"), "wb"))
    log.info(f"Honeypot: {len(hard_exclude)} hard-excluded, "
             f"{len(soft_penalize)} soft-penalized")
    return honeypot_data


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    total_t0 = time.time()

    # Step 1: Load candidates
    candidates = load_candidates(CANDIDATES_PATH)

    # Step 2: Save models
    bi_model = save_models(out_dir)

    # Step 3: Encode JD artifacts
    jd_query_embs, jd_skill_embs, arch_embs = encode_jd_artifacts(bi_model, out_dir)

    # Step 4: Encode candidates + archetype scores + FAISS
    candidate_embs = encode_candidates(bi_model, candidates, out_dir)
    build_faiss_index(candidate_embs, out_dir)

    # Step 5: Skill matching
    skill_lookup = compute_skill_matches(bi_model, candidates, jd_skill_embs, out_dir)

    # Step 6: Calibrate skill threshold
    calibrate_skill_threshold(bi_model, jd_skill_embs, out_dir)

    # Step 7: Extract metadata
    extract_metadata(candidates, out_dir)

    # Step 8: Extract nested data
    extract_nested(candidates, out_dir)

    # Step 9: Honeypot pre-detection
    precompute_honeypot_flags(candidates, skill_lookup, out_dir)

    # Summary
    total_time = time.time() - total_t0
    log.info(f"\n{'='*60}")
    log.info(f"PRECOMPUTE COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    log.info(f"Artifacts saved to: {out_dir}")
    log.info(f"{'='*60}")

    # List artifacts
    for f in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp) / 1024 / 1024
            log.info(f"  {f:40s} {size:8.2f} MB")
        elif os.path.isdir(fp):
            total = sum(
                os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(fp)
                for fn in fns
            ) / 1024 / 1024
            log.info(f"  {f + '/':40s} {total:8.2f} MB")


if __name__ == "__main__":
    main()
