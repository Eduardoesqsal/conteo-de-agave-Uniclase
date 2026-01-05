import os
import torch
from ultralytics import YOLO

# =========================
# CONFIGURACION
# ========================= 
data_yaml = r"C:\Users\GeoSpectral\Desktop\Train Geofield\Geo-Uniclase\data.yaml"
model_path = "yolov9e.pt"

img_size = 1024
epochs = 100
batch_size = 4

project_name = "Geoespectral_modelo"
experiment_name = "GSD-1024-100ep-mAP95"

# Forzar uso de la GPU 0 (no solo "cuda")
device = "0"
print(f"Usando dispositivo: GPU {device}")

# =========================
# ENTRENAMIENTO - MAXIMA PRECISION (mAP50-95)
# =========================
model = YOLO(model_path)

model.train(
    data=data_yaml,
    imgsz=img_size,
    epochs=epochs,
    batch=batch_size,
    project=project_name,
    name=experiment_name,
    device=device,  # <-- Esto ahora fuerza la GPU 0
    exist_ok=True,

    # Optimizacion para precision fina
    optimizer="AdamW",
    lr0=0.0001,
    weight_decay=0.0005,

    # Aumentos suaves
    hsv_h=0.015,
    hsv_s=0.25,
    hsv_v=0.15,
    fliplr=0.5,
    scale=0.15,
    translate=0.05,
    mosaic=0.25,
    mixup=0.0,

    # Estabilidad y convergencia
    warmup_epochs=3,
    cos_lr=True,
    close_mosaic=10,

    # Compatibilidad Windows
    workers=0,

    # Entrenamiento completo
    patience=0,
)

print("ENTRENAMIENTO COMPLETADO - 100 EPOCAS, GPU FORZADA, ENFOCADO EN MAXIMA PRECISION (mAP50-95)")