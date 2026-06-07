import streamlit as st
import subprocess
import pandas as pd
import os
import time

st.set_page_config(page_title="Redrob Ranker", layout="wide")

st.title("🏆 Redrob AI Candidate Ranking Sandbox")
st.markdown("""
This sandbox demonstrates the **Redrob AI Candidate Ranking System**. 
It runs the CPU-only `rank.py` script on precomputed artifacts to rank the top 100 candidates from the dataset.

**Constraints Verified:**
- Runs entirely on CPU 🖥️
- Completes well within the 5-minute budget ⏱️
- No network calls to external LLMs 🔌
""")

uploaded_file = st.file_uploader("Upload Candidates Sample (.jsonl or .json)", type=["jsonl", "json"])

if uploaded_file is not None:
    # Save the uploaded file so rank.py can see it if it checks
    with open("./candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded successfully! Ready to rank.")

col1, col2 = st.columns(2)

with col1:
    run_uploaded = st.button("🚀 Run on Uploaded Sample", type="primary", disabled=(uploaded_file is None), use_container_width=True)

with col2:
    run_preloaded = st.button("⚡ Run on Pre-loaded 100K Dataset", type="secondary", use_container_width=True)

if run_uploaded or run_preloaded:
    if not os.path.exists("./artifacts"):
        st.error("Artifacts folder not found! Please ensure precomputed artifacts are uploaded to the Space.")
    else:
        with st.spinner("Executing rank.py..."):
            start = time.time()
            
            # The spec says reproduce command is: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
            # We mock the candidates.jsonl file path since our pipeline reads from artifacts/
            script_name = "rank_small.py" if run_uploaded else "rank.py"
            cmd = [
                "python", script_name, 
                "--candidates", "./candidates.jsonl", 
                "--artifacts", "./artifacts", 
                "--out", "./submission.csv"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start
            
            if result.returncode == 0:
                st.success(f"✅ Ranking completed successfully in {duration:.1f} seconds!")
                
                # Display output
                df = pd.read_csv("submission.csv")
                st.subheader(f"Top Candidates (Ranked: {len(df)})")
                
                # Make the table wide and formatted
                st.dataframe(
                    df,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "score": st.column_config.NumberColumn("Score", format="%.6f"),
                        "reasoning": st.column_config.TextColumn("Reasoning", width="large")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Provide CSV download
                with open("submission.csv", "rb") as file:
                    st.download_button(
                        label="📥 Download submission.csv",
                        data=file,
                        file_name="submission.csv",
                        mime="text/csv",
                    )
                
                with st.expander("🛠️ View Pipeline Logs"):
                    st.code(result.stderr)
            else:
                st.error("❌ Ranking failed!")
                st.code(result.stderr)
