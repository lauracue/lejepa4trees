# -*- coding: utf-8 -*-

import os
import json
from PIL import Image

image_dir = "./data/correcting_geospatial_data/images"
json_dir = "./data/correcting_geospatial_data/labels_json"
output_dir = "./data/correcting_geospatial_data/cropped_images"

os.makedirs(output_dir, exist_ok=True)

crop_size = 128
half = crop_size // 2

# build class dict ON THE FLY
class_to_idx = {}
next_class_id = 0


def get_class_id(class_name):
    global next_class_id

    if class_name not in class_to_idx:
        class_to_idx[class_name] = next_class_id
        next_class_id += 1

    return class_to_idx[class_name]


for file_name in os.listdir(image_dir):

    if not file_name.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, file_name)
    json_path = os.path.join(json_dir, file_name.replace(".jpg", ".json"))

    if not os.path.exists(json_path):
        continue

    # ---- load image ----
    img = Image.open(image_path)
    width, height = img.size

    # ---- load json ----
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---- class mapping ----
    cls_name = data.get("target_class", None)

    if cls_name is None:
        continue

    cls_id = get_class_id(cls_name)

    # ---- extract bbox ----
    bbox = data.get("tree_gt_bbox_canvas_px", None)

    if bbox is None:
        print(f"No bbox in {file_name}, skipping.")
        continue

    try:
        bx1, by1, bx2, by2 = map(int, bbox)

        # ensure bbox is valid
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(width, bx2)
        by2 = min(height, by2)

        if bx2 <= bx1 or by2 <= by1:
            print(f"Invalid bbox in {file_name}, skipping.")
            continue

        # ---- FIRST: crop image by bbox ----
        bbox_crop = img.crop((bx1, by1, bx2, by2))

        bw, bh = bbox_crop.size

        # ---- SECOND: take centered 128x128 crop ----
        cx = bw // 2
        cy = bh // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # adjust if crop exceeds image borders
        if x2 > bw:
            x2 = bw
            x1 = max(0, bw - crop_size)

        if y2 > bh:
            y2 = bh
            y1 = max(0, bh - crop_size)

        final_crop = bbox_crop.crop((x1, y1, x2, y2))

        # optional:
        # if bbox crop is smaller than 128x128, resize to 128x128
        final_crop = final_crop.resize((128, 128))

        # ---- save ----
        base = os.path.splitext(file_name)[0]
        out_name = f"{base}_cls{cls_id}.jpg"
        out_path = os.path.join(output_dir, out_name)

        final_crop.save(out_path)

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Image can't be cropped: {file_name}")
        print(e)
