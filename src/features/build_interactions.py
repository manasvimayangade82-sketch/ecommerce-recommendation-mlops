import os
import pandas as pd

INPUT_FILE = "data/processed/clean_transactions.csv"
OUTPUT_FILE = "data/processed/user_item_interactions.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["CustomerID"] = df["CustomerID"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)

interactions = (
    df.groupby(["CustomerID", "StockCode"], as_index=False)
    .agg(
        interaction=("Quantity", "sum"),
        purchase_count=("InvoiceNo", "nunique"),
        total_spent=("TotalPrice", "sum")
    )
)

interactions["interaction"] = interactions["interaction"].clip(lower=0)

interactions.to_csv(OUTPUT_FILE, index=False)

print("Interaction dataset created")
print(f"Rows: {len(interactions)}")
print(f"Customers: {interactions['CustomerID'].nunique()}")
print(f"Products: {interactions['StockCode'].nunique()}")
print(f"Average interactions per customer: {len(interactions) / interactions['CustomerID'].nunique():.2f}")
print(f"Saved to: {OUTPUT_FILE}")