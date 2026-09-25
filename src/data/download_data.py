import os
import urllib.request

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
output = "data/raw/online_retail.xlsx"

os.makedirs("data/raw", exist_ok=True)

print("Downloading dataset...")
urllib.request.urlretrieve(url, output)
print(f"Dataset saved to {output}")