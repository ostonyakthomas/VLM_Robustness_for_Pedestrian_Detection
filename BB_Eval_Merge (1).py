#!/usr/bin/env python3
"""
VLM benchmarking pipeline — fixed version with proper incremental aggregation.
"""

import os
import json
import base64
import requests
import re
import traceback
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from datetime import datetime
from PIL import Image
import io

# ------------------ CONFIG ------------------
OPENROUTER_KEY_FILE = "./OpenRouter-2.txt"
try:
    with open(OPENROUTER_KEY_FILE, "r") as f:
        API_KEY = f.read().strip()
except Exception as e:
    raise SystemExit(f"Could not read API key from {OPENROUTER_KEY_FILE}: {e}")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 180

MODELS = [
    "openai/gpt-5",
    "google/gemini-2.5-pro",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4.1-fast",
    "z-ai/glm-4.6v"
]

IMG_ROOT = "./citypersons/"
AGE_GT_DIR = "./labels_citypersons/age/"
GENDER_GT_DIR = "./labels_citypersons/gender/"
OUT_DIR = "./Full_Run/Bounding_Box_Results"
DETECTIONS_DIR = os.path.join(OUT_DIR, "detections")
RAW_DIR = os.path.join(OUT_DIR, "raw_responses")
RESULTS_DIR = os.path.join(OUT_DIR, "results")

os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODE = "all"
ALLOWED_EXTS = (".png", ".jpg", ".jpeg")
IOU_THRES = 0.5
REPEATS = 1

PROMPT = (
    "You are analyzing an image. Respond ONLY with valid JSON.\n"
    "Return a single JSON object with exactly one key: \"people_bboxes\".\n"
    "Detect each pedestrian separately — do NOT combine multiple people into one box.\n"
    "Each bounding box must tightly enclose the visible extent of that person within the image.\n"
    "Bounding boxes must be formatted as [x1, y1, x2, y2] using integer pixel coordinates.\n"
    "If no people are visible, return {\"people_bboxes\": []}.\n"
)


# ---------------- Utility functions ----------------

def sanitize_model_name(model_str):
    """Return a filesystem-friendly model name."""
    name = model_str.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name

def safe_json_extract(text):
    """Try to extract JSON from text robustly."""
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    
    def find_first_balanced(s, open_ch, close_ch):
        stack = 0
        start = None
        for i, ch in enumerate(s):
            if ch == open_ch:
                if start is None:
                    start = i
                stack += 1
            elif ch == close_ch and start is not None:
                stack -= 1
                if stack == 0:
                    return s[start:i+1]
        return None

    cand = find_first_balanced(text, "{", "}")
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass
    cand = find_first_balanced(text, "[", "]")
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass
    try:
        alt = text.replace("'", '"')
        return json.loads(alt)
    except Exception:
        pass
    return None

def cal_iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    try:
        if boxA is None or boxB is None:
            return 0.0
        a = [float(x) for x in boxA]
        b = [float(x) for x in boxB]
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])

        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH

        boxAArea = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        boxBArea = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0.0:
            return 0.0
        return interArea / denom
    except Exception:
        return 0.0

def read_gt_boxes(gt_path):
    """Read GT text file, return list of dicts: {'label': label_or_None, 'box': [x1,y1,x2,y2]}"""
    boxes = []
    if not os.path.isfile(gt_path):
        return boxes
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 4:
                coords_tokens = parts[-4:]
                try:
                    coords = [int(round(float(x))) for x in coords_tokens]
                except Exception:
                    continue
                label = None
                if len(parts) >= 5:
                    lead = parts[:-4]
                    if len(lead) >= 1:
                        try:
                            label = int(float(lead[0]))
                        except Exception:
                            label = None
                boxes.append({'label': label, 'box': coords})
    return boxes

def match_gender_to_age_boxes(age_boxes, gender_boxes):
    """Return matched list of persons with age + optional gender."""
    out = []
    gender_by_coords = {tuple(g['box']): g['label'] for g in gender_boxes}
    used_gender_idx = set()
    for a in age_boxes:
        a_box = a['box']
        g_label = None
        key = tuple(a_box)
        if key in gender_by_coords:
            g_label = gender_by_coords[key]
        else:
            best_iou = 0.0
            best_idx = None
            for i, g in enumerate(gender_boxes):
                if i in used_gender_idx:
                    continue
                iou = cal_iou(a_box, g['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx is not None and best_iou >= 0.5: #Ensure match, changed from 0.9
                g_label = gender_boxes[best_idx]['label']
                used_gender_idx.add(best_idx)
        out.append({'age': a.get('label'), 'gender': g_label, 'box': a_box})
    return out

def ensure_dir(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def save_raw_response(run_id, main_image_folder, model, raw_text):
    subdir = os.path.join(RAW_DIR, f"run_{run_id}", main_image_folder)
    ensure_dir(subdir)
    fname = os.path.join(subdir, f"{sanitize_model_name(model)}_raw.txt")
    header = f"MODEL: {model}\nSTATUS: saved\nTIMESTAMP: {datetime.utcnow().isoformat()}Z\n\n"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(header + (raw_text if isinstance(raw_text, str) else str(raw_text)))
    except Exception as e:
        print(f"[ERROR] Saving raw response to {fname}: {e}")

def save_detection_json(run_id, main_image_folder, model, detections):
    subdir = os.path.join(DETECTIONS_DIR, f"run_{run_id}", main_image_folder)
    ensure_dir(subdir)
    fname = os.path.join(subdir, f"{sanitize_model_name(model)}_detections.json")
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({'people_bboxes': detections}, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Saving detections to {fname}: {e}")

def load_and_resize_base64(image_path, max_dim=1024):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1:
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def query_vlm_for_image(run_id, model, image_path, main_image_folder):
    """Query VLM via OpenRouter. Returns list of [x1,y1,x2,y2] integer boxes."""
    try:
        img_b64 = load_and_resize_base64(image_path)
        image_payload = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Expert pedestrian detector."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        image_payload
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        print(f"[QUERY] {model} -> {main_image_folder}/{os.path.basename(image_path)}")
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        try:
            raw_text = f"HTTP_STATUS: {r.status_code}\n\n{r.text}"
        except Exception:
            raw_text = f"HTTP_STATUS: {r.status_code}\n\n{str(r.content)}"

        save_raw_response(run_id, main_image_folder, model, raw_text)

        if r.status_code == 402 or "insufficient credits" in r.text.lower() or "quota exceeded" in r.text.lower():
            raise SystemExit(f"[ERROR] Credits exhausted while querying {model} for {image_path}")

        if r.status_code != 200:
            print(f"  [QUERY] Non-200 HTTP {r.status_code} for {model}. Returning []")
            return []

        parsed = None
        try:
            j = r.json()
            choices = j.get("choices") or j.get("outputs") or []
            if isinstance(choices, list) and choices:
                c0 = choices[0]
                if isinstance(c0, dict):
                    content = (
                        c0.get("message", {}).get("content")
                        or c0.get("text")
                        or c0.get("content")
                    )
                    parsed = safe_json_extract(content) if content else None
                else:
                    parsed = safe_json_extract(json.dumps(c0))
            else:
                parsed = safe_json_extract(json.dumps(j))
        except Exception:
            parsed = safe_json_extract(raw_text)

        if not parsed:
            parsed = safe_json_extract(raw_text)

        if not parsed:
            print(f"  [PARSE] No JSON parsed from {model} for {image_path}. Treating as no detections.")
            return []

        out_boxes = []

        if isinstance(parsed, dict) and "people_bboxes" in parsed:
            candidate_lists = [parsed["people_bboxes"]]
        elif isinstance(parsed, dict):
            candidate_lists = []
            possible_keys = ["bboxes", "boxes", "person_boxes", "people", "detections"]
            for k in possible_keys:
                if k in parsed and isinstance(parsed[k], list):
                    candidate_lists.append(parsed[k])
            for v in parsed.values():
                if isinstance(v, list):
                    candidate_lists.append(v)
                elif isinstance(v, dict):
                    for sub in v.values():
                        if isinstance(sub, list):
                            candidate_lists.append(sub)
        elif isinstance(parsed, list):
            candidate_lists = [parsed]
        else:
            candidate_lists = []

        for cand in candidate_lists:
            if isinstance(cand, list):
                for b in cand:
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        try:
                            out_boxes.append([int(round(float(x))) for x in b])
                        except Exception:
                            continue

        save_detection_json(run_id, main_image_folder, model, out_boxes)
        print(f"  [DETECT] Model {model} returned {len(out_boxes)} boxes")
        return out_boxes

    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout querying {model} for {image_path}")
        return []
    except Exception as e:
        print(f"  [ERROR] Exception querying {model} for {image_path}: {e}")
        traceback.print_exc(limit=1)
        return []


# ----------------- Image listing -----------------

def get_all_image_folders_and_files(img_root):
    """Returns list of (main_folder, image_filename)"""
    pairs = []
    if not os.path.isdir(img_root):
        print(f"[IMG] Image root not found: {img_root}")
        return pairs
    for entry in sorted(os.listdir(img_root)):
        entry_path = os.path.join(img_root, entry)
        if os.path.isdir(entry_path):
            files = sorted([f for f in os.listdir(entry_path) if f.lower().endswith(ALLOWED_EXTS)])
            for f in files:
                pairs.append((entry, f))
        else:
            if entry.lower().endswith(ALLOWED_EXTS):
                pairs.append(('', entry))
    print(f"[IMG] Found {len(pairs)} images under {img_root}")
    return pairs

def get_first_image_overall(img_root):
    pairs = get_all_image_folders_and_files(img_root)
    return pairs[0] if pairs else (None, None)

def get_first_image_per_folder(img_root):
    out = []
    if not os.path.isdir(img_root):
        return out
    for entry in sorted(os.listdir(img_root)):
        entry_path = os.path.join(img_root, entry)
        if os.path.isdir(entry_path):
            files = sorted([f for f in os.listdir(entry_path) if f.lower().endswith(ALLOWED_EXTS)])
            if files:
                out.append((entry, files[0]))
        else:
            if entry.lower().endswith(ALLOWED_EXTS):
                out.append(('', entry))
    return out


# ----------------- Evaluation / Matching -----------------

def evaluate_image_models(run_id, main_folder, image_name, image_path, models):
    """Evaluate all models on one image - using UNION of age and gender boxes."""
    base = os.path.splitext(image_name)[0]
    age_gt_path = os.path.join(AGE_GT_DIR, f"{base}.txt")
    gender_gt_path = os.path.join(GENDER_GT_DIR, f"{base}.txt")

    age_boxes = read_gt_boxes(age_gt_path)
    gender_boxes = read_gt_boxes(gender_gt_path) if os.path.exists(gender_gt_path) else []
    
    # NEW: Create union of all unique people
    gt_persons = merge_age_and_gender_boxes(age_boxes, gender_boxes)

    per_model_results = {}

    for model in models:
        detections = query_vlm_for_image(run_id, model, image_path, main_folder)
        used_det_indices = set()
        per_gt_list = []

        for gt_idx, gt in enumerate(gt_persons):
            gt_box = gt['box']
            best_iou = 0.0
            best_det = None
            best_det_idx = None
            for di, det_box in enumerate(detections):
                if di in used_det_indices:
                    continue
                current_iou = cal_iou(gt_box, det_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_det = det_box
                    best_det_idx = di
            matched = False
            if best_det_idx is not None and best_iou >= IOU_THRES:
                matched = True
                used_det_indices.add(best_det_idx)
            per_gt_list.append({
                'GT_idx': gt_idx,
                'GT_box': gt_box,
                'GT_age': gt['age'],
                'GT_gender': gt['gender'],
                'Detected': matched,
                'Matched_IoU': round(best_iou, 4),
                'Best_Det_Box': best_det if best_det is not None else None
            })
        per_model_results[model] = {'detections': detections, 'per_gt': per_gt_list}
    return gt_persons, per_model_results


def merge_age_and_gender_boxes(age_boxes, gender_boxes):
    """
    Merge age and gender boxes into a unified list of all unique people.
    Uses IoU ≥ 0.5 to identify when boxes refer to the same person.
    """
    all_persons = []
    
    # Start with all age boxes
    for age_box in age_boxes:
        # Try to find matching gender box
        best_iou = 0.0
        best_gender_label = None
        
        for gender_box in gender_boxes:
            iou = cal_iou(
                [age_box['label'], *age_box['box']], 
                [gender_box['label'], *gender_box['box']]
            )
            if iou > best_iou:
                best_iou = iou
                best_gender_label = gender_box['label']
        
        # If IoU ≥ 0.5, this age box has a matching gender box
        gender_label = best_gender_label if best_iou >= 0.5 else None
        
        all_persons.append({
            'age': age_box['label'],
            'gender': gender_label,
            'box': age_box['box'],
            'source': 'age' if gender_label is None else 'both'
        })
    
    # Now add gender boxes that DON'T match any age box
    for gender_box in gender_boxes:
        # Check if this gender box already matched an age box
        already_matched = False
        for age_box in age_boxes:
            iou = cal_iou(
                [age_box['label'], *age_box['box']], 
                [gender_box['label'], *gender_box['box']]
            )
            if iou >= 0.5:
                already_matched = True
                break
        
        # If not matched, add as gender-only person
        if not already_matched:
            all_persons.append({
                'age': None,  # No age label
                'gender': gender_box['label'],
                'box': gender_box['box'],
                'source': 'gender'
            })
    
    return all_persons

# ----------------- Aggregation & Stats -----------------

def aggregate_stats(all_query_results):
    """Aggregate statistics per query (model per GT entry)."""
    df = pd.DataFrame(all_query_results)
    if df.empty:
        empty_stats = pd.DataFrame(columns=[
            'Model','Attribute','Class','Total','Success','Miss_Rate','Z_Stat','P_Value','Significant'
        ])
        return df, empty_stats

    df['GT_Gender'] = pd.to_numeric(df['GT_Gender'], errors='coerce')
    df['GT_Age'] = pd.to_numeric(df['GT_Age'], errors='coerce')
    df['Detected'] = pd.to_numeric(df['Detected'], errors='coerce').fillna(0).astype(int)

    def calc_attribute_stats(df, attribute, mapping):
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
                success = int((cls_df['Detected'] == 1).sum())
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
    age_map = {0: 'Adult', 1: 'Child'}

    gender_stats = calc_attribute_stats(df, 'GT_Gender', gender_map)
    age_stats = calc_attribute_stats(df, 'GT_Age', age_map)

    stats = pd.concat([gender_stats, age_stats], ignore_index=True) if (not gender_stats.empty or not age_stats.empty) else pd.DataFrame()
    return df, stats


# ----------------- Pipeline -----------------

def run_pipeline_once(run_id):
    print(f"\n===== RUN {run_id} START =====")

    if MODE == "first_overall":
        m, im = get_first_image_overall(IMG_ROOT)
        pairs = [(m, im)] if m is not None else []
    elif MODE == "first_per_folder":
        pairs = get_first_image_per_folder(IMG_ROOT)
    elif MODE == "all":
        pairs = get_all_image_folders_and_files(IMG_ROOT)
    else:
        raise SystemExit(f"Unknown MODE: {MODE}")

    print(f"[PIPELINE] MODE={MODE}. Images to process: {len(pairs)}")
    total_images = len(pairs)

    checkpoint_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    ensure_dir(checkpoint_dir)

    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
    checkpoint_csv_file = os.path.join(checkpoint_dir, f"vlm_checkpoint_run{run_id}.csv")

    # Load checkpoint
    if os.path.exists(checkpoint_csv_file):
        print(f"[CHECKPOINT] Loading existing checkpoint CSV")
        df_prev = pd.read_csv(checkpoint_csv_file)
        all_query_results = df_prev.to_dict(orient="records")
        checkpoint = set()
        for r in all_query_results:
            checkpoint.add((r['Main_Image_Folder'], r['Image'], r['Model']))
        print(f"[CHECKPOINT] Resuming from {len(checkpoint)} already processed queries")
    else:
        all_query_results = []
        checkpoint = set()

    for idx, (main_folder, image_name) in enumerate(pairs, start=1):
        main_folder_name = main_folder if main_folder else os.path.splitext(image_name)[0]

        # Skip if all models already processed
        if all((main_folder_name, image_name, model) in checkpoint for model in MODELS):
            print(f"[SKIP] Already processed: {main_folder_name}/{image_name}")
            continue

        image_path = os.path.join(IMG_ROOT, main_folder, image_name) if main_folder else os.path.join(IMG_ROOT, image_name)
        if not os.path.exists(image_path):
            print(f"[SKIP] Image not found: {image_path}")
            continue

        print(f"\n[IMAGE] Processing image {idx}/{total_images}: {main_folder_name}/{image_name}")

        gt_persons, per_model_results = evaluate_image_models(run_id, main_folder_name, image_name, image_path, MODELS)

        # Save per-image JSON
        matches_dir = os.path.join(DETECTIONS_DIR, f"run_{run_id}", main_folder_name)
        ensure_dir(matches_dir)
        matches_fname = os.path.join(matches_dir, f"{os.path.splitext(image_name)[0]}_matches.json")
        try:
            with open(matches_fname, "w", encoding="utf-8") as f:
                json.dump({
                    'main_image': main_folder_name,
                    'image_file': image_name,
                    'gt_persons': gt_persons,
                    'models': per_model_results
                }, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Saving matches JSON to {matches_fname}: {e}")

        # Flatten per-query results
        for model in MODELS:
            info = per_model_results[model]
            for gt_entry in info['per_gt']:
                entry = {
                    'Main_Image_Folder': main_folder_name,
                    'Image': image_name,
                    'Model': model,
                    'GT_Age': gt_entry['GT_age'],
                    'GT_Gender': gt_entry['GT_gender'],
                    'Detected': int(gt_entry['Detected'])
                }
                all_query_results.append(entry)
                checkpoint.add((main_folder_name, image_name, model))

        # Incremental save
        try:
            temp_per_person_df, temp_stats_df = aggregate_stats(all_query_results)
            temp_per_person_csv = os.path.join(checkpoint_dir, f"vlm_per_person_results_run{run_id}_partial.csv")
            temp_stats_csv = os.path.join(checkpoint_dir, f"vlm_person_detection_stats_run{run_id}_partial.csv")
            temp_per_person_df.to_csv(temp_per_person_csv, index=False)
            temp_stats_df.to_csv(temp_stats_csv, index=False)

            # Save checkpoint CSV (this is the key for resuming)
            pd.DataFrame(all_query_results).to_csv(checkpoint_csv_file, index=False)
        except Exception as e:
            print(f"[ERROR] Saving incremental CSVs: {e}")

    # Final aggregation & save
    per_person_df, stats_df = aggregate_stats(all_query_results)
    run_out_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    ensure_dir(run_out_dir)
    per_person_csv = os.path.join(run_out_dir, f"vlm_per_person_results_run{run_id}.csv")
    stats_csv = os.path.join(run_out_dir, f"vlm_person_detection_stats_run{run_id}.csv")
    try:
        per_person_df.to_csv(per_person_csv, index=False)
        stats_df.to_csv(stats_csv, index=False)
        print(f"[SAVE_CSV] Per-person saved -> {per_person_csv}")
        print(f"[SAVE_CSV] Stats saved -> {stats_csv}")
    except Exception as e:
        print(f"[ERROR] Saving final CSVs: {e}")

    print(f"===== RUN {run_id} COMPLETE =====")
    return per_person_df, stats_df


# ----------------- MAIN -----------------

if __name__ == "__main__":
    print("Starting VLM benchmarking pipeline.")
    for r in range(1, REPEATS + 1):
        try:
            run_df, run_stats = run_pipeline_once(r)
        except Exception as e:
            print(f"[ERROR] Run {r} failed: {e}")
            traceback.print_exc(limit=1)
    print("All runs completed.")
