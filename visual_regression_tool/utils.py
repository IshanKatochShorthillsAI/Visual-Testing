import json
import time
from pathlib import Path
from datetime import datetime
import base64
import hashlib
import re

from playwright.sync_api import sync_playwright, Error as PlaywrightError
from PIL import Image
import cv2
from skimage.metrics import structural_similarity
import numpy as np


# --- Image Comparison Functions (from previous version) ---
def crop_to_common_area(img1, img2):
    """
    Crops two OpenCV images to their smallest common dimensions.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    min_h = min(h1, h2)
    min_w = min(w1, w2)
    cropped1 = img1[0:min_h, 0:min_w]
    cropped2 = img2[0:min_h, 0:min_w]
    return cropped1, cropped2


def compare_images_ssim(img_baseline_path, img_current_path):
    baseline_img = cv2.imread(str(img_baseline_path))
    current_img = cv2.imread(str(img_current_path))
    if baseline_img is None or current_img is None:
        return 0.0, Image.new("RGB", (100, 30), color="red")
    # Crop images to common size before comparison
    baseline_img, current_img = crop_to_common_area(baseline_img, current_img)
    baseline_gray = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(baseline_gray, current_gray, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return score, Image.fromarray(thresh)


# --- Robust Page Loading (from previous version) ---
def robust_scroll_to_bottom(page):
    last_height = page.evaluate("document.body.scrollHeight")
    while True:
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        page.wait_for_timeout(500)
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)


# --- History Management Functions ---
HISTORY_FILE = Path("test_history.json")


def load_history():
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(new_result):
    history = load_history()
    history.insert(0, new_result)  # Add new result to the top
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


# --- HTML Report Generation ---
def image_to_base64(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None


def url_to_filename(url):
    # Remove protocol and replace unsafe characters
    filename = re.sub(r"[^a-zA-Z0-9]", "_", url)
    return filename


def generate_html_report(result):
    baseline_b64 = image_to_base64(result["baseline_path"])
    current_b64 = image_to_base64(result["current_path"])
    diff_b64 = image_to_base64(result["diff_path"])

    html_template = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Visual Regression Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; background-color: #f9f9f9; }}
            h1, h2 {{ color: #333; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 2em; }}
            .card {{ background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .card h2 {{ margin-top: 0; }}
            .card img {{ max-width: 100%; border: 1px solid #ccc; }}
            .metadata {{ margin-bottom: 2em; }}
            .fail {{ color: #D32F2F; }}
        </style>
    </head>
    <body>
        <h1>Visual Regression Report</h1>
        <div class=\"metadata\">
            <p><strong>URL:</strong> <a href=\"{result['url']}\">{result['url']}</a></p>
            <p><strong>Timestamp:</strong> {result['timestamp']}</p>
            <p class=\"fail\"><strong>Result:</strong> {result['status']} (Score: {result['score']:.4f})</p>
        </div>
        <div class=\"container\">
            <div class=\"card\"><h2>Baseline</h2><img src=\"data:image/png;base64,{baseline_b64}\"></div>
            <div class=\"card\"><h2>Current</h2><img src=\"data:image/png;base64,{current_b64}\"></div>
            <div class=\"card\"><h2>Difference</h2><img src=\"data:image/png;base64,{diff_b64}\"></div>
        </div>
    </body>
    </html>
    """
    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    safe_name = result.get("key", "report")
    report_filename = (
        report_path
        / f"report_{safe_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    )
    with open(report_filename, "w") as f:
        f.write(html_template)
    return report_filename


def compare_images_features(img_baseline_path, img_current_path):
    baseline_img = cv2.imread(str(img_baseline_path))
    current_img = cv2.imread(str(img_current_path))
    if baseline_img is None or current_img is None:
        return 0.0, Image.new("RGB", (100, 30), color="red")
    # Crop images to common size before comparison
    baseline_img, current_img = crop_to_common_area(baseline_img, current_img)
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(baseline_img, None)
    kp2, des2 = orb.detectAndCompute(current_img, None)
    if des1 is None or des2 is None or len(kp1) == 0:
        return 0.0, Image.new("RGB", (100, 30), color="red")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_score = len(matches) / len(kp1) if len(kp1) > 0 else 0
    match_visualization = cv2.drawMatches(
        baseline_img,
        kp1,
        current_img,
        kp2,
        matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return match_score, Image.fromarray(
        cv2.cvtColor(match_visualization, cv2.COLOR_BGR2RGB)
    )


# --- The Main Test Runner Function (MODIFIED) ---
def run_single_test(url, config):
    safe_name = url_to_filename(url)
    base_dir = Path("visual_baselines")
    base_dir.mkdir(exist_ok=True)

    result = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "key": safe_name,
        "status": "pending",
        "score": 0.0,
        "baseline_path": str(base_dir / f"{safe_name}_baseline.png"),
        "current_path": str(base_dir / f"{safe_name}_current.png"),
        "diff_path": str(base_dir / f"{safe_name}_diff.png"),
    }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.add_init_script("document.body.style.caretColor = 'transparent'")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)
            robust_scroll_to_bottom(page)

            mask_locators = []
            if config.get("mask_selectors"):
                for selector in config["mask_selectors"]:
                    if selector:
                        mask_locators.append(page.locator(selector))

            page.screenshot(
                path=result["current_path"], full_page=True, mask=mask_locators
            )
    except PlaywrightError as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        return result

    baseline_path = Path(result["baseline_path"])
    if not baseline_path.exists():
        Path(result["current_path"]).rename(baseline_path)
        result["status"] = "new_baseline"
        return result

    # --- NEW: Call the correct comparison function based on config ---
    algorithm = config.get("algorithm", "Structural Similarity (SSIM)")
    threshold = config.get("threshold", 0.999)
    if algorithm == "Feature Matching (ORB)":
        score, diff_img = compare_images_features(
            result["baseline_path"], result["current_path"]
        )
        result["score"] = score * 100  # Convert to percentage
        pass_condition = result["score"] >= threshold
    else:  # Default to SSIM
        score, diff_img = compare_images_ssim(
            result["baseline_path"], result["current_path"]
        )
        result["score"] = score
        pass_condition = score >= threshold

    if pass_condition:
        result["status"] = "pass"
        Path(result["current_path"]).unlink()
    else:
        result["status"] = "fail"
        diff_img.save(result["diff_path"])

    return result
