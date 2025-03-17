import gradio as gr 
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np
import time

QDRANT_HOST = "10.84.0.7"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120


collection_name='image_db'
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT,timeout=QDRANT_TIMEOUT)
print("[INFO] Client created...")

#loading the model 
print("[INFO] Loading the model...")
model_name = "/home/reifr1z/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

def process_text(image):
    processed_img = processor(text=None, images=image, return_tensors="pt")['pixel_values']
    img_embeddings = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    hits = client.query_points(
        collection_name=collection_name,
        query=img_embeddings,
        search_params=models.SearchParams(hnsw_ef=256, exact=True),
        limit=5,
    )

    # Zugriff auf die Liste der ScoredPoint-Objekte
    scored_points = hits.points  # Annahme: hits hat ein Attribut 'points'
    
    # Sortieren der Ergebnisse nach Score (absteigend)
    scored_points.sort(key=lambda x: x.score, reverse=True)
    
    # Überprüfen, ob es überhaupt Ergebnisse gibt
    if not scored_points:
        return None
    
    # Das Bild mit dem höchsten Score auswählen
    best_scored_point = scored_points[0]
    
    # Extrahieren des Payloads
    payload = best_scored_point.payload
    #print(best_scored_point.score)
    # Extrahieren der Image Size und Pixel Liste aus dem Payload
    img_size = tuple(payload['img_size'])
    pixel_lst = payload['pixel_lst']
    
    # Überprüfen, ob pixel_lst eine flache Liste ist und umwandeln in Tupel
    if isinstance(pixel_lst[0], list):
        pixel_lst = [tuple(pixel) for pixel in pixel_lst]
    elif isinstance(pixel_lst[0], int):
        pixel_lst = [tuple(pixel_lst[i:i+3]) for i in range(0, len(pixel_lst), 3)]
    
    # Create an image from pixel data
    new_image = Image.new("RGB", img_size)
    new_image.putdata(pixel_lst)
    
    return [new_image]

    
# Gradio Interface
iface = gr.Interface(
    title="Reverse Image Search Engine",
    fn=process_text,
    inputs=gr.Image(label="Input Image"),
    outputs=gr.Gallery(label="Relevant Images"),  
)

iface.launch(server_name='0.0.0.0',server_port=8504)