#!/usr/bin/env python3
"""
VLM person detection pipeline for cropped images - fixed version.
"""

import os
import json
import base64
import requests
import pandas as pd
import time
import traceback
from tqdm import tqdm
from statsmodels.stats.proportion import proportions_ztest

# =========================================================
# ---------------------- CONFIG ---------------------------
# =========================================================
with open("./OpenRouter-2.txt", "r") as f:
    API_KEY = f.read().strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


MODELS = [
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4.1-fast",
    "z-ai/glm-4.6v"
]

CROP_DIR = "./Cropped_CityPersons/"
OUT_DIR = "./Full_Run/Cropped_Results"
RAW_DIR = os.path.join(OUT_DIR, "raw_responses")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

ALLOWED_EXTS = (".png", ".jpg", ".jpeg")
REQUEST_TIMEOUT = 180

PROMPT = (
    "You are analyzing an image.\n"
    "Respond ONLY with valid JSON. EXACTLY one JSON object. Base your answer STRICTLY on visible evidence. "
    "Before deciding, internally verify at least one visible visual cue (e.g., human silhouette, limbs, head, clothing, reflection).\n"
    "{\"person_present\": 1} if a person is visible.\n"
    "{\"person_present\": 0} if no person is visible.\n"
    "{\"person_present\": -1} if the image is not visible to you.\n"
    "Do NOT guess.\n"
    "Nothing else.\n"
)

# =========================================================
# --------------------- UTILITIES -------------------------
# =========================================================
def safe_json_parse(text):
    """Try to extract JSON from text."""
    try:
        return json.loads(text)
    except:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except:
            return None

def save_checkpoint(results):
    """Save current progress to disk."""
    try:
        df = pd.DataFrame(results)
        checkpoint_csv = os.path.join(OUT_DIR, "vlm_person_detection_checkpoint.csv")
        df.to_csv(checkpoint_csv, index=False)
        
        # Also compute and save aggregated stats
        df_agg, stats = aggregate_stats(results)
        df_agg.to_csv(os.path.join(OUT_DIR, "agg_rows_checkpoint.csv"), index=False)
        stats.to_csv(os.path.join(OUT_DIR, "agg_stats_checkpoint.csv"), index=False)
    except Exception as e:
        print(f"[ERROR] Checkpoint save failed: {e}")
        traceback.print_exc(limit=1)

def query_person(model, img_path, attempt=1):
    """Query VLM for person detection in cropped image."""
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        # OpenRouter-compatible image payload
        image_payload = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Expert pedestrian detector."},
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    image_payload
                ]}
            ]
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Prepare raw response directory
        img_name_only = os.path.splitext(os.path.basename(img_path))[0]
        main_image_name = os.path.basename(os.path.dirname(img_path))
        img_raw_dir = os.path.join(RAW_DIR, main_image_name, img_name_only)
        os.makedirs(img_raw_dir, exist_ok=True)

        print(f"[QUERY] {model} -> {main_image_name}/{img_name_only}")
        start_time = time.time()
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        elapsed = time.time() - start_time
        print(f"  [RESPONSE] {r.status_code} in {elapsed:.2f}s")

        # Save raw response
        rawfname = os.path.join(img_raw_dir, f"{model.split('/')[-1]}_raw.txt")
        with open(rawfname, "w", encoding="utf-8") as f:
            f.write(f"HTTP_STATUS: {r.status_code}\n\n{r.text}")

        # Check for credit exhaustion
        if r.status_code == 402 or "insufficient credits" in r.text.lower() or "quota exceeded" in r.text.lower():
            raise RuntimeError("OpenRouter: Out of credits detected. Stopping script.")

        if r.status_code != 200:
            print(f"  [ERROR] Non-200 response, returning None")
            return None

        # Parse response
        j = r.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = safe_json_parse(content)
        
        if parsed and "person_present" in parsed:
            result = int(parsed["person_present"])
            print(f"  [RESULT] person_present={result}")
            return result
        
        print(f"  [PARSE] Failed to parse person_present, returning 0")
        return 0

    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] Attempt {attempt}")
        if attempt < 2:
            time.sleep(1)
            return query_person(model, img_path, attempt + 1)
        return None
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc(limit=1)
        return None

def extract_age_gender_from_filename(img_name):
    """Extract age and gender labels from filename."""
    n = img_name.lower()
    age = None
    gender = None
    
    if "child" in n:
        age = 1
    elif "adult" in n:
        age = 0
    
    if "unknown-gender" in n:
        gender = None
    elif "female" in n or "woman" in n:
        gender = 1
    elif "male" in n or "man" in n:
        gender = 0
    
    return age, gender

def get_all_images(crop_dir):
    """Get all image paths from crop directory."""
    all_imgs = []
    for folder in sorted(os.listdir(crop_dir)):
        fpath = os.path.join(crop_dir, folder)
        if not os.path.isdir(fpath):
            continue
        for f in sorted(os.listdir(fpath)):
            if f.lower().endswith(ALLOWED_EXTS):
                all_imgs.append((folder, f))
    return all_imgs

def count_all_images(crop_dir):
    """Count total images in crop directory."""
    total = 0
    for folder in os.listdir(crop_dir):
        folder_path = os.path.join(crop_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.lower().endswith(ALLOWED_EXTS):
                total += 1
    return total

# =========================================================
# ---------------------- AGGREGATION ----------------------
# =========================================================
def aggregate_stats(results):
    """Aggregate statistics from results list."""
    df = pd.DataFrame(results)
    if df.empty:
        empty_stats = pd.DataFrame(columns=[
            'Model','Attribute','Class','Total','Success','Miss_Rate','Z_Stat','P_Value','Significant'
        ])
        return df, empty_stats

    # Ensure numeric types
    df['GT_Gender'] = pd.to_numeric(df['GT_Gender'], errors='coerce')
    df['GT_Age'] = pd.to_numeric(df['GT_Age'], errors='coerce')
    df['Pred'] = pd.to_numeric(df['Pred'], errors='coerce').fillna(0).astype(int)

    def calc_attribute_stats(df, attribute, mapping):
        """Calculate statistics for a given attribute."""
        results = []
        for model in sorted(df['Model'].unique()):
            model_df = df[df['Model'] == model]
            model_df_known = model_df[model_df[attribute].notna()]
            if model_df_known.empty:
                continue

            counts = {}
            for cls in sorted(model_df_known[attribute].unique()):
                cls_df = model_df_known[model_df_known[attribute] == cls]
                total = len(cls_df)
                success = int((cls_df['Pred'] == 1).sum())
                miss_rate = 1 - (success / total) if total > 0 else None
                counts[cls] = {'total': total, 'success': success, 'miss_rate': miss_rate}

            z_stat = None
            p_val = None
            if len(counts) == 2:
                cls_vals = sorted(counts.keys())
                succ = [counts[c]['success'] for c in cls_vals]
                tot = [counts[c]['total'] for c in cls_vals]
                if tot[0] > 0 and tot[1] > 0:
                    try:
                        z_stat, p_val = proportions_ztest(succ, tot, alternative='two-sided')
                    except Exception:
                        z_stat, p_val = None, None

            for cls, v in counts.items():
                results.append({
                    'Model': model,
                    'Attribute': attribute,
                    'Class': mapping.get(cls, str(cls)),
                    'Total': v['total'],
                    'Success': v['success'],
                    'Miss_Rate': v['miss_rate'],
                    'Z_Stat': z_stat,
                    'P_Value': p_val
                })

        resdf = pd.DataFrame(results)
        if 'P_Value' not in resdf.columns:
            resdf['P_Value'] = None
        resdf['Significant'] = resdf['P_Value'].apply(lambda p: 'Yes' if p is not None and p < 0.05 else 'No')
        return resdf

    gender_map = {0.0: 'Male', 1.0: 'Female'}
    age_map = {0.0: 'Adult', 1.0: 'Child'}

    gender_stats = calc_attribute_stats(df, 'GT_Gender', gender_map)
    age_stats = calc_attribute_stats(df, 'GT_Age', age_map)

    stats = pd.concat([gender_stats, age_stats], ignore_index=True) if (not gender_stats.empty or not age_stats.empty) else pd.DataFrame()
    return df, stats

# =========================================================
# ------------------------ MAIN ---------------------------
# =========================================================

# Load checkpoint if it exists
checkpoint_csv = os.path.join(OUT_DIR, "vlm_person_detection_checkpoint.csv")
if os.path.exists(checkpoint_csv):
    print("[INFO] Found existing checkpoint. Resuming from last saved state...")
    df_checkpoint = pd.read_csv(checkpoint_csv)
    processed_set = set(zip(
        df_checkpoint["Main_Image_Folder"],
        df_checkpoint["Crop_File"],
        df_checkpoint["Model"]
    ))
    results = df_checkpoint.to_dict(orient="records")
    print(f"[INFO] Loaded {len(results)} previous results, {len(processed_set)} processed queries")
else:
    print("[INFO] No checkpoint found. Starting fresh.")
    processed_set = set()
    results = []

print("Counting total cropped images...")
TOTAL_IMAGES = count_all_images(CROP_DIR)
print(f"Total cropped images detected: {TOTAL_IMAGES}\n")

all_images = get_all_images(CROP_DIR)
total_queries = len(all_images) * len(MODELS)
completed_queries = len(processed_set)

print(f"Total queries: {total_queries}")
print(f"Completed queries: {completed_queries}")
print(f"Remaining queries: {total_queries - completed_queries}\n")

try:
    for folder, img_name in tqdm(all_images, desc="Images", ncols=80):
        img_path = os.path.join(CROP_DIR, folder, img_name)
        gt_age, gt_gender = extract_age_gender_from_filename(img_name)

        for model in MODELS:
            # Skip if already processed
            if (folder, img_name, model) in processed_set:
                continue

            try:
                pred = query_person(model, img_path)
            except RuntimeError as e:
                print(f"\n{e}\nCheckpointing and exiting...")
                save_checkpoint(results)
                raise SystemExit("Stopped due to credit exhaustion.")

            # Add result
            results.append({
                "Main_Image_Folder": folder,
                "Crop_File": img_name,
                "GT_Age": gt_age,
                "GT_Gender": gt_gender,
                "Model": model,
                "Pred": pred
            })

            # Mark as processed and save checkpoint
            processed_set.add((folder, img_name, model))
            save_checkpoint(results)

except KeyboardInterrupt:
    print("\nInterrupted by user. Saving checkpoint...")
    save_checkpoint(results)
    raise

# =========================================================
# ------------------------ FINAL SAVE --------------------
# =========================================================
df = pd.DataFrame(results)
out_csv = os.path.join(OUT_DIR, "vlm_person_detection_all.csv")
out_json = os.path.join(OUT_DIR, "vlm_person_detection_all.json")
df.to_csv(out_csv, index=False)
df.to_json(out_json, orient="records", indent=4)

df_agg, stats = aggregate_stats(results)
df_agg.to_csv(os.path.join(OUT_DIR, "agg_rows.csv"), index=False)
stats.to_csv(os.path.join(OUT_DIR, "agg_stats.csv"), index=False)

print(f"\nDONE. Saved {len(df)} rows.")
print("CSV:", out_csv)
print("JSON:", out_json)
print("\nAggregation completed. Stats summary:")
print(stats.head(10))
