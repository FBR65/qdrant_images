import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np
import secrets
import uuid

QDRANT_HOST = "10.84.0.7"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120


collection_name='sw_image_db'
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT,timeout=QDRANT_TIMEOUT)

def generate_point_id():
    # Generiere eine UUID
    uuid_value = uuid.uuid4().hex

    # Ersetze bestimmte Ziffern durch zufällige Werte
    modified_uuid = ''.join(
        (hex((int(c, 16) ^ secrets.randbits(4) & 15 >> int(c) // 4))[2:] if c in '018' else c)
        for c in uuid_value
    )

    return str(modified_uuid)


print("[INFO] Client created...")

################### Dataset Loading ###################
image_dataset = []  

root_dir = "dataset"  

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        #look only for image files with jpeg extension
        if  file.endswith(".jpg"):  
            image_path = os.path.join(subdir, file)
            try:
                image = Image.open(image_path).convert('L')  
                image_dataset.append(image)  
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")  



################### Loading the CLIP model ###################
print("[INFO] Loading the model...")
model_name = "/home/reifr1z/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

###################----Creating a qdrant collection----######################
print("[INFO] Creating qdrant data collection...")

collection_distances = ["COSINE","EUCLID","DOT","MANHATTAN"]

for distances in collection_distances:
    collection = collection_name + '_' + str(distances)

    match distances:
        case "COSINE":
            distance = models.Distance.COSINE
        case "EUCLID":
            distance = models.Distance.EUCLID
        case "DOT":
            distance = models.Distance.DOT
        case "Manhattan":
            distance = models.Distance.MANHATTAN

    if not client.collection_exists(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=512, distance=distance),
        
        )

###################----creating records/vectors ----######################
print("[INFO] Creating a data collection...")
records = []
for idx, sample in tqdm(enumerate(image_dataset), total=len(image_dataset)):
    processed_img = processor(text=None, images = sample, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    img_px = list(sample.getdata())
    img_size = sample.size 
    print("[INFO] Uploading data record to data collection...")
    point_id = generate_point_id()
    for distances in collection_distances:
        collection = collection_name + '_' + str(distances)
        client.upload_points(
            collection_name = collection,
            points=[
                models.PointStruct(id=point_id, vector=img_embds, payload={"pixel_lst":img_px, "img_size": img_size})
                ]
            )


print("[INFO] Successfully uploaded data records to data collection!")