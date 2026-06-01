import requests
import json
from pathlib import Path

output_dir = Path("data/raw/rng")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading QRNG data from ANU...")

# API configuration
QRN_URL = "https://api.quantumnumbers.anu.edu.au/"
QRN_KEY = "zb2iPbif8PrVMB13Tto55gJNjz7P0ih64nnYgnH6"

params = {
    "length": 1024,
    "type": "uint8"
}

headers = {"x-api-key": QRN_KEY}

response = requests.get(QRN_URL, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()

    output_file = output_dir / "anu_sample.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    if data.get('success'):
        print(f"✅ Saved: {output_file}")
        print(f"📊 Downloaded {len(data['data'])} quantum numbers")
    else:
        print(f"❌ API error: {data.get('message')}")
else:
    print(f"❌ HTTP {response.status_code}")
    print(response.text)