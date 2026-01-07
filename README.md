# conteo-de-agave-Uniclase
Entrenamiento de inteligencia artificial enfoque Geoespacial
🌱 Conteo de Agave – Enfoque Geoespacial (Uniclase)


Este proyecto implementa un flujo de detección y conteo automático de agave a partir de ortomosaicos georreferenciados, utilizando inteligencia artificial y técnicas de visión por computadora geoespacial.

El sistema está basado en el modelo YOLOv9e, entrenado para detectar agaves individuales en imágenes aéreas, y aplicado mediante un pipeline que divide el ortomosaico en ventanas (tiles) para procesar grandes extensiones de terreno de forma eficiente.

Las detecciones se filtran espacial y geométricamente para evitar duplicados y ruido, y posteriormente se georreferencian, generando resultados listos para su análisis en sistemas de información geográfica (SIG) como QGIS o ArcGIS.

El proyecto está orientado a aplicaciones en:

agricultura de precisión

monitoreo de cultivos

análisis territorial

<img width="1746" height="967" alt="Captura de pantalla 2025-12-07 225510" src="https://github.com/user-attachments/assets/e657c3b6-e04d-4caf-9c38-d095b4c99024" />


