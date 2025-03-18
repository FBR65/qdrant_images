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

collection_name = 'sw_image_db'
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
print("[INFO] Client created...")

# Loading the model 
print("[INFO] Loading the model...")
model_name = "/home/reifr1z/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

def process_text(image):
    if image is None:
        return [], [], [], []

    # Convert the image to a PIL Image object
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    image = image.convert('L')
    
    processed_img = processor(text=None, images=image, return_tensors="pt")['pixel_values']
    img_embeddings = model.get_image_features(processed_img).detach().numpy().tolist()[0]

    collection_distances = ["COSINE", "EUCLID", "DOT", "MANHATTAN"]
    galleries = {}

    for distance in collection_distances:
        collection = collection_name + '_' + distance

        hits = client.query_points(
            collection_name=collection,
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
            galleries[distance] = []
            continue
        
        gallery_images = []
        for scored_point in scored_points:
            payload = scored_point.payload
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
            
            # Label für das Bild mit dem Score
            label = f"Score: {scored_point.score:.2f}"
            gallery_images.append((new_image, label))
        
        galleries[distance] = gallery_images
    
    return [galleries[distance] for distance in collection_distances]

# Gradio Interface
with gr.Blocks() as iface:
    gr.HTML("<center><h1> Reverse Image Search Engine </h1></center>")
    with gr.Row():
        input_image = gr.Image(label="Input Image")
    with gr.Row():
        search_button = gr.Button("Suche", variant='primary')
        clear_button = gr.ClearButton(value="Löschen", variant="secondary")
    with gr.Row():
        with gr.Column():
            gallery_cosine = gr.Gallery(label="COSINE")
            gallery_dot = gr.Gallery(label="DOT")
        with gr.Column():
            gallery_euclid = gr.Gallery(label="EUCLID")
            gallery_manhattan = gr.Gallery(label="MANHATTAN")
    
    search_button.click(fn=process_text, inputs=input_image, outputs=[gallery_cosine, gallery_dot, gallery_euclid, gallery_manhattan])
    clear_button.add([input_image, gallery_cosine, gallery_dot, gallery_euclid, gallery_manhattan])    

iface.launch(server_name='0.0.0.0', server_port=8504)