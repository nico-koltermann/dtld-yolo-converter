from ultralytics import YOLO
import os

################################################################
# Config
################################################################

# root of your dataset
DATA_SET = "./data"  

# 'yolov8n.pt' = nano 
# 'yolov8s.pt' = small, etc. {n,m,l,x}
YOLO_MODEL = "yolov8n.pt"

IMAGE_SIZE = 640

################################################################
################################################################

def ensure_folder():

    dataset_dir = os.path.abspath(DATA_SET)

    train_images = os.path.join(dataset_dir, "train/images")
    train_labels = os.path.join(dataset_dir, "train/labels")
    val_images = os.path.join(dataset_dir, "val/images")
    val_labels = os.path.join(dataset_dir, "val/labels")

    print("Train images:", train_images)
    print("Train labels:", train_labels)
    print("Validation images:", val_images)
    print("Validation labels:", val_labels)

    return train_images, train_labels, val_images, val_labels

def train():
    train_images, train_labels, val_images, val_labels = ensure_folder()

    dataset_yaml = os.path.join(dataset_dir, "traffic_dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"""\
    train: {train_images}
    val: {val_images}

    nc: 3  # number of classes (red, yellow, green)
    names: ['red', 'yellow', 'green']
    """)

    print(f"Dataset YAML saved at {dataset_yaml}")

    model = YOLO(YOLO_MODEL)

    # --- Train ---
    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=IMAGE_SIZE,
        batch=8,
        name="traffic_light_yolov8",
        project="runs/train"
    )

    print("Training started!")


if __name__ == "__main__":
    train()
