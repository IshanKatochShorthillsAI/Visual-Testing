import streamlit as st
import pandas as pd
import utils  # Import our utility file

st.set_page_config(page_title="Test History", layout="wide")
st.title("📜 Test Run History")

history_data = utils.load_history()

if not history_data:
    st.info(
        "No test history found. Run some tests on the main page to see results here."
    )
else:
    df = pd.DataFrame(history_data)

    # --- Data Cleaning and Formatting for Display ---
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["score"] = df["score"].apply(lambda x: f"{x:.4f}")

    # --- Display Metrics ---
    total_runs = len(df)
    pass_rate = (
        (df["status"].value_counts().get("pass", 0) / total_runs) * 100
        if total_runs > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", total_runs)
    col2.metric("Pass Rate", f"{pass_rate:.1f}%")

    # --- Display DataFrame ---
    st.dataframe(
        df[["timestamp", "status", "url", "score"]],
        use_container_width=True,
        hide_index=True,
    )
