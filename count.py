import cv2
import numpy as np
from ultralytics import YOLO
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from rasterio.windows import Window
import torch

# ==========================
# CONFIG MAXIMA DETECCION
# ==========================
MODEL_PATH = r"C:\Users\GeoSpectral\Desktop\Train Geofield\Geo-Uniclase\Geoespectral_modelo\GSD-FIX-1024\weights\best.pt"
ORTOMOSAICO_PATH = r"C:\Users\GeoSpectral\Desktop\Train Geofield\mosaicos\pruebas\ANA MARÍA - EL FRIJOL 08-08-2025 ORT.tif"

TILE_SIZE = 1024
CORE_SIZE = 896
STRIDE = CORE_SIZE

CONF_THRESHOLD = 0.03
IOU_THRESHOLD = 0.4

MIN_AREA = 500
MAX_AREA = 60000

COUNT_FILE = "conteo_maximo.txt"
OUTPUT_SHP = "agaves_maximo.shp"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================
print("Cargando modelo...")
model = YOLO(MODEL_PATH)
print(f"Usando dispositivo: {DEVICE}")

with rasterio.open(ORTOMOSAICO_PATH) as src:
    transform = src.transform
    crs = src.crs
    width, height = src.width, src.height
    print(f"Dimensiones: {width} x {height}")

all_centers = []
tile_count = 0

offset = (TILE_SIZE - CORE_SIZE) // 2

print("Procesando en modo MAXIMA DETECCION...")
with rasterio.open(ORTOMOSAICO_PATH) as src:
    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):

            window = Window(
                max(x - offset, 0),
                max(y - offset, 0),
                TILE_SIZE,
                TILE_SIZE
            )

            tile = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
            tile = np.transpose(tile, (1, 2, 0))
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

            tile_count += 1

            results = model(
                tile,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                half=True,
                verbose=False
            )

            for r in results:
                if r.boxes is None:
                    continue

                for box, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy()
                ):
                    if int(cls) != 0:
                        continue

                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)

                    if area < MIN_AREA or area > MAX_AREA:
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if not (
                        offset < cx < offset + CORE_SIZE and
                        offset < cy < offset + CORE_SIZE
                    ):
                        continue

                    global_x = int(x + cx - offset)
                    global_y = int(y + cy - offset)

                    all_centers.append((global_x, global_y))

print(f"Tiles procesados: {tile_count}")
print(f"TOTAL AGAVES DETECTADOS: {len(all_centers)}")

# ==========================
# CONTEO
# ==========================
with open(COUNT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Agave_0: {len(all_centers)}\n")
    f.write(f"\nTOTAL AGAVES DETECTADOS: {len(all_centers)}\n")

print(f"Conteo guardado: {COUNT_FILE}")

# ==========================
# SHAPEFILE
# ==========================
points = [
    Point(rasterio.transform.xy(transform, cy, cx, offset="center"))
    for cx, cy in all_centers
]

gdf = gpd.GeoDataFrame(
    {"id": range(1, len(points) + 1)},
    geometry=points,
    crs=crs
)

gdf.to_file(OUTPUT_SHP)
print(f"Shapefile guardado: {OUTPUT_SHP}")
print("Proceso completado.")
