import os
from ultralytics import YOLO

# ================================
# 1️⃣ Parameters
# ================================
train_data_yaml = "us_data.yaml"
val_dir = "./val_data"
output_dir = "runs/predict_val"
exp_name = "ultrasound_cancer_detector"
img_size = 640
batch_size = 4
epochs = 100
device = 0  # GPU ID

# ================================
# 2️⃣ Initialize YOLO model
# ================================
model = YOLO("yolov8s.pt")  # Can be replaced with yolov8n/m/l/x.pt

# ================================
# 3️⃣ Train YOLO model
# ================================
print("Starting YOLO model training...")
train_results = model.train(
    data=train_data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    name=exp_name,
    device=device,
    save=True  # Will automatically save best.pt
)
print("Training completed. Best weights path:", os.path.join("runs", "train", exp_name, "weights", "best.pt"))

# ================================
# 4️⃣ Validation set prediction (batch)
# ================================
print(f"Starting batch prediction on validation set {val_dir}...")
results = model(val_dir, save=True, project=output_dir)
print(f"Prediction completed. Results saved to {output_dir}")
