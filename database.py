import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForZeroShotImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np

client = QdrantClient("localhost", port=6333)
print("[INFO] Client created...")

################### Dataset Loading ###################
image_dataset = []  

root_dir = "dataset/cane"  

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        #look only for image files with jpeg extension
        if  file.endswith(".jpeg"):  
            image_path = os.path.join(subdir, file)
            try:
                image = Image.open(image_path)  
                image_dataset.append(image)  
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")  



################### Loading the CLIP model ###################
print("[INFO] Loading the model...")
model_name = "./models/openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

###################----Creating a qdrant collection----######################
print("[INFO] Creating qdrant data collection...")
client.create_collection(
    collection_name="animals_img_db",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),

)

###################----creating records/vectors ----######################
print("[INFO] Creating a data collection...")
records = []
for idx, sample in tqdm(enumerate(image_dataset), total=len(image_dataset)):
    processed_img = processor(text=None, images = sample, return_tensors="pt")['pixel_values']
    img_embds = model.get_image_features(processed_img).detach().numpy().tolist()[0]
    img_px = list(sample.getdata())
    img_size = sample.size 
    records.append(models.Record(id=idx, vector=img_embds, payload={"pixel_lst":img_px, "img_size": img_size}))


#uploading the records to client
print("[INFO] Uploading data records to data collection...")
#It's better to upload chunks of data to the VectorDB 
for i in range(30,len(records), 30):
    print(f"finished {i}")
    client.upload_records(
        collection_name="animals_img_db",
        records=records[i-30:i],
    )

print("[INFO] Successfully uploaded data records to data collection!")