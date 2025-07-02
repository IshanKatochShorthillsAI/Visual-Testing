import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_image_comparison import image_comparison
import utils  # Import our new utility file
from PIL import Image

st.set_page_config(page_title="Visual Regression Testing", layout="wide")
st.title("🚀 Visual Regression Testing")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- NEW: Add Feature Matching to the algorithm selector ---
    comparison_algo = st.radio(
        "Comparison Algorithm",
        ("Structural Similarity (SSIM)", "Feature Matching (ORB)"),
        index=0,  # Default to SSIM
        help="SSIM is good for layout. Feature Matching is good for content shifts.",
    )

    # --- Batch Testing Input ---
    st.subheader("Test Input")
    uploaded_file = st.file_uploader(
        "Upload a list of URLs (TXT or CSV, one URL per line)", type=["txt", "csv"]
    )
    url_input = st.text_input("Or enter a single URL:")

    # --- Test Parameters ---
    st.subheader("Test Parameters")
    if comparison_algo == "Structural Similarity (SSIM)":
        threshold = st.slider(
            "Similarity Threshold",
            0.90,
            1.0,
            0.999,
            0.001,
            "%.3f",
            help="The minimum structural similarity required for a test to pass. 1.0 is a perfect match.",
        )
    else:
        threshold = st.slider(
            "Match Threshold (%)",
            0,
            100,
            85,
            1,
            help="Minimum percentage of features from the baseline that must be matched in the current image to pass.",
        )
    mask_selectors_str = st.text_input(
        "CSS Selectors to Mask (comma-separated)",
        help="e.g., .hero-carousel, #ad-banner",
    )

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def request_stop():
    st.session_state.stop_requested = True


st.button("🛑 Stop Batch Run", on_click=request_stop, use_container_width=True)

if st.button("Run Visual Tests", use_container_width=True):
    st.session_state.stop_requested = False  # Reset stop flag at start
    urls_to_test = []
    if uploaded_file is not None:
        try:
            # Read URLs from uploaded file
            df = pd.read_csv(uploaded_file, header=None)
            urls_to_test = df[0].tolist()
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif url_input:
        urls_to_test = [url_input]

    if not urls_to_test:
        st.warning("Please provide at least one URL to test.")
    else:
        st.info(f"Starting test run for {len(urls_to_test)} URL(s)...")
        progress_bar = st.progress(0)
        results_placeholder = st.container()
        summary = []

        config = {
            "algorithm": comparison_algo,
            "threshold": threshold,
            "mask_selectors": [s.strip() for s in mask_selectors_str.split(",")],
        }

        for i, url in enumerate(urls_to_test):
            if st.session_state.stop_requested:
                st.warning("Batch run stopped by user.")
                break
            with st.spinner(f"Testing {url}..."):
                result = utils.run_single_test(url, config)
                utils.save_history(result)
                summary.append(result)

            # Display individual result in an expander
            with results_placeholder.expander(
                f"{'✅' if result['status'] == 'pass' else '❌'} {result['url']}",
                expanded=True,
            ):
                # --- Warn if dimensions differ ---
                try:
                    baseline_pil = Image.open(result["baseline_path"])
                    current_pil = Image.open(result["current_path"])
                    if baseline_pil.size != current_pil.size:
                        st.warning(
                            f"⚠️ **Dimension Mismatch:** Baseline is {baseline_pil.size} but Current is {current_pil.size}. "
                            f"Comparison was performed on the common area of ({min(baseline_pil.width, current_pil.width)} x {min(baseline_pil.height, current_pil.height)})."
                        )
                except FileNotFoundError:
                    pass  # This is handled by the 'new_baseline' status

                if result["status"] == "pass":
                    st.success(f"PASS ({comparison_algo} Score: {result['score']:.4f})")
                elif result["status"] == "new_baseline":
                    st.info("New baseline created.")
                    st.image(result["baseline_path"])
                elif result["status"] == "error":
                    st.error(
                        f"Test Error: {result.get('error_message', 'Unknown error')}"
                    )
                else:  # Fail
                    st.error(f"FAIL ({comparison_algo} Score: {result['score']:.4f})")

                    if comparison_algo == "Feature Matching (ORB)":
                        st.subheader("Feature Match Visualization")
                        st.image(
                            result["diff_path"],
                            caption="Lines connect matched features between baseline (left) and current (right).",
                        )
                    else:
                        image_comparison(
                            img1=result["baseline_path"],
                            img2=result["current_path"],
                            label1="Baseline",
                            label2="Current",
                            width=700,
                            starting_position=50,
                            show_labels=True,
                            make_responsive=True,
                        )

                    # --- Sharable HTML Report ---
                    report_path = utils.generate_html_report(result)
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download HTML Report",
                            data=f,
                            file_name=report_path.name,
                            mime="text/html",
                        )

            progress_bar.progress((i + 1) / len(urls_to_test))

        st.header("📋 Run Summary")
        summary_df = pd.DataFrame(summary)[["url", "status", "score"]]
        st.dataframe(summary_df, use_container_width=True)
