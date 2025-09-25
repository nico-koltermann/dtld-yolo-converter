import glob
import json
import os

import cv2
import shutil

################################################################
# Config
################################################################

DATA_SETS = ['Bochum', 'Dortmund']

DATASET_BASE_PATH = "./dtld"

TRAIN_VAL_SPLIT = 0.8 # [%]

TARGET_DIR = "./data"

CLASS_MAP = {"red": 0, "yellow": 1, "green": 2}

################################################################
################################################################

def get_data_path():

    matches = glob.glob(DATASET_BASE_PATH)

    if not matches:
        raise FileNotFoundError(f"No folder found matching: {DATASET_BASE_PATH}")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple matches found: {matches}")
    else:
        return matches[0]


def get_data_set_info():
    data = { "images": [] }
    for DS in DATA_SETS:
        with open(os.path.join(DATA_PATH, 'DTLD_Labels_v2.0', 'v2.0', f'{DS}.json')) as f:
            new_data = json.load(f)
            data["images"].extend(new_data["images"])
    return data

def load_labels(label_data):
    labels = []
    for label in label_data:
        x, y, w, h = label["x"], label["y"], label["w"], label["h"]
        attrs = label["attributes"]
    
        labels.append({
            "position": (x, y, w, h),
            "attributes": attrs
        })

    return labels

def copy_images(data_set_info, start_ind: int, end_ind: int, target_dir: str):
    indices = list(range(start_ind, end_ind))
    for index, img_info in zip(indices, data_set_info['images'][start_ind:end_ind]):
        
        labels = load_labels(img_info["labels"])
        file_name = str(index).zfill(5)

        source_path = os.path.join(DATA_PATH, img_info['image_path'])
        target_path = os.path.join(TARGET_DIR, target_dir, "images", f"{file_name}.tiff")
        
        shutil.copyfile(source_path, target_path)

        img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

        # Image dimensions for normalization
        h, w = img.shape[:2]

        lines = []
        for obj in labels:
            x, y, bw, bh = obj["position"]

            # Convert to YOLO format (normalized)
            cls = CLASS_MAP.get(obj["attributes"]["state"].lower(), -1)
            if cls == -1:
                continue  # skip unknown labels

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            w_norm = bw / w
            h_norm = bh / h

            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(os.path.join(TARGET_DIR, target_dir, "labels", f"{file_name}.txt"), "w") as f:
            f.write("\n".join(lines))

def ensure_folder_structure(target_dir):
    for split in ["train", "val"]:
        images_path = os.path.join(target_dir, split, "images")
        labels_path = os.path.join(target_dir, split, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

if __name__ == "__main__":

    DATA_PATH = get_data_path()
    print("DATA_PATH =", DATA_PATH)

    data_set_info = get_data_set_info()

    length = len(data_set_info['images'])
    train_end = int(length * TRAIN_VAL_SPLIT)

    print(f"Total images: {length}")
    print(f"Training from {0} to {train_end}, val until {length}")

    ensure_folder_structure(TARGET_DIR)

    copy_images(data_set_info, 0, train_end, "train")
    copy_images(data_set_info, train_end, length, "val")

    print("Data copied successfully!")
