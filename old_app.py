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
model_name = "/home/reifr1z/models/openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

def process_text(image):
    processed_img = processor(text=None, images = image, return_tensors="pt")['pixel_values']
    img_embeddings = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    hits = client.search(
        collection_name=collection_name,
        query_vector=img_embeddings,
        limit=5,
    )

    images = []
    for hit in hits:
        img_size = tuple(hit.payload['img_size'])
        pixel_lst = hit.payload['pixel_lst']
        
        time.sleep(2)
        # Create an image from pixel data
        new_image = Image.new("RGB", img_size)
        new_image.putdata(list(map(lambda x: tuple(x), pixel_lst)))
        images.append(new_image)

    return images

# Gradio Interface
iface = gr.Interface(
    title="Reverse Image Search Engine",
    fn=process_text,
    inputs=gr.Image(label="Input Image"),
    outputs=gr.Gallery(label="Relevant Images"),  
)

iface.launch(server_name='0.0.0.0',server_port=8504)