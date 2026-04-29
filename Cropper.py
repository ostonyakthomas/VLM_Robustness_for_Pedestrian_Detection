import os
import cv2

# ===================== CONFIG ======================
IMG_DIR = "./citypersons/"
AGE_DIR = "./labels_citypersons/age/"
GENDER_DIR = "./labels_citypersons/gender/"
OUT_DIR = "./Cropped_CityPersons/"
PADDING = 20

os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================

def load_gt(gt_path):
    """Load GT bounding boxes as [label, x1, y1, x2, y2]."""
    boxes = []
    if not os.path.isfile(gt_path):
        return boxes
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            label = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[1:])
            boxes.append([label, x1, y1, x2, y2])
    return boxes

def crop_with_padding(img, box, pad):
    """Crop bounding box with padding and clip to image bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)
    return img[y1p:y2p, x1p:x2p]

def cal_iou(boxA, boxB):
    """Compute IoU between two boxes [label, x1, y1, x2, y2]."""
    _, x1A, y1A, x2A, y2A = boxA
    _, x1B, y1B, x2B, y2B = boxB
    
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    if interArea <= 0:
        return 0.0
    
    boxAArea = (x2A - x1A) * (y2A - y1A)
    boxBArea = (x2B - x1B) * (y2B - y1B)
    
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def merge_age_and_gender_boxes(age_boxes, gender_boxes):
    """
    Merge age and gender boxes into unified list of all unique people.
    Returns list of dicts: {'age': label_or_None, 'gender': label_or_None, 'box': [x1,y1,x2,y2]}
    """
    all_persons = []
    matched_gender_indices = set()
    
    # Process all age boxes and match with gender when possible
    for age_box in age_boxes:
        best_iou = 0.0
        best_gender_label = None
        best_gender_idx = None
        
        for g_idx, gender_box in enumerate(gender_boxes):
            if g_idx in matched_gender_indices:
                continue
            iou = cal_iou(age_box, gender_box)
            if iou > best_iou:
                best_iou = iou
                best_gender_label = gender_box[0]
                best_gender_idx = g_idx
        
        # If IoU ≥ 0.5, this age box has a matching gender box
        if best_iou >= 0.5:
            gender_label = best_gender_label
            matched_gender_indices.add(best_gender_idx)
        else:
            gender_label = None
        
        all_persons.append({
            'age': age_box[0],
            'gender': gender_label,
            'box': age_box[1:]
        })
    
    # Add gender boxes that didn't match any age box
    for g_idx, gender_box in enumerate(gender_boxes):
        if g_idx not in matched_gender_indices:
            all_persons.append({
                'age': None,  # No age label
                'gender': gender_box[0],
                'box': gender_box[1:]
            })
    
    return all_persons

# ================= MAIN LOOP =======================

total_processed = 0
total_people = 0

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    
    img_path = os.path.join(IMG_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    
    age_path = os.path.join(AGE_DIR, f"{base}.txt")
    gender_path = os.path.join(GENDER_DIR, f"{base}.txt")
    
    age_boxes = load_gt(age_path)
    gender_boxes = load_gt(gender_path)
    
    # Merge to get all unique people (union)
    all_persons = merge_age_and_gender_boxes(age_boxes, gender_boxes)
    
    if not all_persons:
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not open image: {img_path}")
        continue
    
    out_folder = os.path.join(OUT_DIR, base)
    os.makedirs(out_folder, exist_ok=True)
    
    for idx, person in enumerate(all_persons):
        crop = crop_with_padding(img, person['box'], PADDING)
        
        # Age label
        if person['age'] is None:
            age_label = "unknown-age"
        elif person['age'] == 1:
            age_label = "child"
        else:
            age_label = "adult"
        
        # Gender label
        if person['gender'] is None:
            gender_text = "unknown-gender"
        elif person['gender'] == 1:
            gender_text = "woman"
        else:
            gender_text = "man"
        
        # Filename
        out_path = os.path.join(
            out_folder,
            f"person_{idx}_{age_label}_{gender_text}.jpg"
        )
        cv2.imwrite(out_path, crop)
    
    total_people += len(all_persons)
    total_processed += 1
    print(f"[{total_processed}] Saved {len(all_persons)} crops for {img_name} → {out_folder}")

print(f"\n✅ DONE — Processed {total_processed} images, created {total_people} crops")
print(f"   Age boxes only: count separately if needed")
print(f"   Gender boxes only: count separately if needed")
print(f"   Both age + gender: count separately if needed")
