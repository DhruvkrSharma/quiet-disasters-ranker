import streamlit as st
import subprocess
import pandas as pd
import os
import time

st.set_page_config(page_title="Redrob Ranker", layout="wide", page_icon="🏆")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #07090f !important;
    color: #e2e4ed !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
footer { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ══════════════════════════════════
   CENTERED COLUMN — the ONE truth
   ══════════════════════════════════ */
.center-col {
    max-width: 560px;
    margin: 0 auto;
    padding: 0 24px;
}

/* ══════════════════════════════════
   HERO
   ══════════════════════════════════ */
.hero {
    background: #07090f;
    border-bottom: 1px solid #10141f;
    padding: 72px 24px 56px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -160px; right: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,75,43,0.09) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -100px; left: 20%;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.hero-inner {
    max-width: 680px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: #4dda7a;
    background: rgba(77,218,122,0.07);
    border: 1px solid rgba(77,218,122,0.18);
    border-radius: 999px;
    padding: 6px 13px;
    margin-bottom: 28px;
    text-transform: uppercase;
}
.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #4dda7a;
    animation: blink 2.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.06;
    color: #edeef5;
    letter-spacing: -0.025em;
    margin-bottom: 18px;
}
.hero-title em { color: #FF4B2B; font-style: normal; }

.hero-sub {
    font-size: 14.5px;
    color: #5a6080;
    line-height: 1.7;
    max-width: 480px;
    margin: 0 auto 36px;
}
.hero-sub code {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    background: #10141f;
    border: 1px solid #1a2030;
    border-radius: 5px;
    padding: 1px 6px;
    color: #FF4B2B;
}

.chip-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.07em;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid;
    white-space: nowrap;
    width: auto;
    min-width: unset;
}
.chip-a { border-color: #1a2e40; color: #4da8da; background: rgba(77,168,218,0.05); }
.chip-b { border-color: #1a2e1a; color: #4dda7a; background: rgba(77,218,122,0.05); }
.chip-c { border-color: #2e1a1a; color: #da8a4d; background: rgba(218,138,77,0.05); }

/* ══════════════════════════════════
   MAIN CONTENT COLUMN
   ══════════════════════════════════ */
.main-content {
    max-width: 560px;
    margin: 0 auto;
    padding: 48px 24px 64px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ── Card ── */
.card {
    background: #0c0f1a;
    border: 1px solid #13182a;
    border-radius: 16px;
    padding: 24px;
}
.card-header {
    margin-bottom: 16px;
}
.card-title {
    font-size: 13px;
    font-weight: 700;
    color: #c8cce0;
    letter-spacing: 0.01em;
    margin-bottom: 3px;
}
.card-hint {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #2e3450;
    letter-spacing: 0.06em;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #07090f !important;
    border: 1.5px dashed #1a2035 !important;
    border-radius: 12px !important;
    padding: 0 !important;
    transition: border-color 0.2s;
    min-height: 180px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,75,43,0.4) !important;
}
[data-testid="stFileUploader"] section {
    padding: 32px 20px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    width: 100% !important;
}
[data-testid="stFileUploader"] label {
    color: #3a4060 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    text-align: center !important;
}
[data-testid="stFileUploader"] small {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: #252a3a !important;
}

/* ── Success / error alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    margin-top: 12px !important;
    margin-bottom: 0 !important;
}

/* ── Action buttons — stacked, fit-content, centered ── */
.action-stack {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    border-radius: 9px !important;
    transition: all 0.16s ease !important;
    cursor: pointer !important;
    /* key: fit the label, don't stretch */
    width: 100% !important;
    padding: 12px 28px !important;
    border: none !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #FF4B2B, #FF6840) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(255,75,43,0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(255,75,43,0.42) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    color: #4a5270 !important;
    border: 1px solid #13182a !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #FF4B2B !important;
    color: #e2e4ed !important;
    background: rgba(255,75,43,0.04) !important;
}

/* ── Divider between stacked buttons ── */
.btn-divider {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #1e2235;
    letter-spacing: 0.1em;
    text-align: center;
}

/* ══════════════════════════════════
   RESULTS
   ══════════════════════════════════ */
.result-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #07090f;
    border: 1px solid #13182a;
    border-radius: 8px;
    margin-bottom: 14px;
}
.result-ok {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #4dda7a;
    background: rgba(77,218,122,0.08);
    border: 1px solid rgba(77,218,122,0.2);
    border-radius: 5px;
    padding: 2px 8px;
    letter-spacing: 0.07em;
}
.result-time {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #4dda7a;
}
.result-n {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #252a3a;
    margin-left: auto;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid #13182a !important;
    overflow: hidden !important;
}
[data-testid="stDownloadButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    background: transparent !important;
    border: 1px solid #13182a !important;
    color: #4a5270 !important;
    border-radius: 8px !important;
    padding: 9px 18px !important;
    transition: all 0.15s !important;
    margin-top: 12px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #FF4B2B !important;
    color: #e2e4ed !important;
}
[data-testid="stExpander"] {
    background: #07090f !important;
    border: 1px solid #13182a !important;
    border-radius: 9px !important;
    margin-top: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: #2e3450 !important;
}
[data-testid="stSpinner"] p { color: #FF4B2B !important; font-size: 13px !important; }

@media (max-width: 600px) {
    .main-content { padding: 32px 16px 48px; }
    .hero { padding: 48px 16px 40px; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div style="display:flex;justify-content:center;margin-bottom:28px">
      <span class="status-badge"><span class="status-dot"></span>Sandbox Active</span>
    </div>
    <div class="hero-title">Candidate <em>Ranking</em> System</div>
    <div class="hero-sub">
      Run the <code>rank.py</code> pipeline on precomputed artifacts
      to surface the top 100 candidates from your dataset —
      entirely on CPU, no network required.
    </div>
    <div class="chip-row">
      <span class="chip chip-a">🖥 CPU only · No GPU</span>
      <span class="chip chip-b">⚡ &lt; 5 min · Budget compliant</span>
      <span class="chip chip-c">🔌 No network · Fully offline</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# UPLOAD CARD
# ══════════════════════════════════════════════════════
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-title">Upload Candidates Sample</div>
    <div class="card-hint">.JSONL &nbsp;or&nbsp; .JSON</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Streamlit uploader rendered INSIDE the card area
# (We close the card-header in HTML above; the uploader sits below via st.*)
uploaded_file = st.file_uploader(
    "Drag & drop or click to upload",
    type=["jsonl", "json"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    with open("./candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅  **{uploaded_file.name}** · {uploaded_file.size:,} bytes ready")

# ══════════════════════════════════════════════════════
# EXECUTE CARD
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="card" style="margin-top:16px">
  <div class="card-header">
    <div class="card-title">Execute Pipeline</div>
    <div class="card-hint">Select a data source and run</div>
  </div>
</div>
""", unsafe_allow_html=True)

run_uploaded = st.button(
    "▷  Run on Uploaded Sample",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
)

st.markdown('<div class="btn-divider">or</div>', unsafe_allow_html=True)

run_preloaded = st.button(
    "⚡  Run on Pre-loaded 100K Dataset",
    type="secondary",
    use_container_width=True,
)

st.markdown('</div>', unsafe_allow_html=True)  # /main-content

# ══════════════════════════════════════════════════════
# EXECUTION LOGIC  (untouched)
# ══════════════════════════════════════════════════════
if run_uploaded or run_preloaded:
    if not os.path.exists("./artifacts"):
        st.markdown('<div class="main-content" style="padding-top:0">', unsafe_allow_html=True)
        st.error("❌  Artifacts folder not found! Ensure precomputed artifacts are uploaded to the Space.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Executing rank.py — hang tight..."):
            start = time.time()
            script_name = "rank_small.py" if run_uploaded else "rank.py"
            cmd = [
                "python", script_name,
                "--candidates", "./candidates.jsonl",
                "--artifacts", "./artifacts",
                "--out",        "./submission.csv",
            ]
            result   = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start

            if result.returncode == 0:
                st.session_state.run_success = True
                st.session_state.run_failed  = False
                st.session_state.duration    = duration
                st.session_state.stderr      = result.stderr
            else:
                st.session_state.run_success = False
                st.session_state.run_failed  = True
                st.session_state.stderr      = result.stderr

# ══════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════
if st.session_state.get("run_success", False) and os.path.exists("submission.csv"):
    st.markdown('<div class="main-content" style="padding-top:0"><div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><div class="card-title">Results</div></div>', unsafe_allow_html=True)

    df = pd.read_csv("submission.csv")
    st.markdown(f"""
    <div class="result-meta">
      <span class="result-ok">✓ COMPLETE</span>
      <span class="result-time">⏱ {st.session_state.duration:.2f}s</span>
      <span class="result-n">{len(df):,} candidates</span>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df,
        column_config={
            "rank":      st.column_config.NumberColumn("Rank",      format="%d"),
            "score":     st.column_config.NumberColumn("Score",     format="%.6f"),
            "reasoning": st.column_config.TextColumn("Reasoning",   width="large"),
        },
        use_container_width=True,
        hide_index=True,
    )

    with open("submission.csv", "rb") as f:
        st.download_button(
            label="📥  Download submission.csv",
            data=f,
            file_name="submission.csv",
            mime="text/csv",
        )

    with st.expander("🛠  View Pipeline Logs"):
        st.code(st.session_state.stderr, language="bash")

    st.markdown('</div></div>', unsafe_allow_html=True)

elif st.session_state.get("run_failed", False):
    st.markdown('<div class="main-content" style="padding-top:0"><div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><div class="card-title">Pipeline Error</div></div>', unsafe_allow_html=True)
    st.error("❌  Execution failed. Check logs below.")
    st.code(st.session_state.stderr, language="bash")
    st.markdown('</div></div>', unsafe_allow_html=True)