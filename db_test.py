import requests

# Beispiel für eine manuelle Anfrage
url = "http://10.84.0.7:6333/collections/logo_img_db/points"
headers = {"Content-Type": "application/json"}
data = {
    "points": [
        {
            "id": 0,
            "vector": [0.1, 0.2, 0.3],  # Beispiel-Vektor
            "payload": {"key": "value"}  # Beispiel-Payload
        }
    ]
}

response = requests.put(url, headers=headers, json=data, timeout=60)
print(f"[DEBUG] Response: {response.status_code}")
print(f"[DEBUG] Response Body: {response.text}")