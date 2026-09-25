import os
import pandas as pd

INPUT_FILE = "data/processed/clean_transactions.csv"
OUTPUT_FILE = "data/processed/product_features.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["StockCode"] = df["StockCode"].astype(str)

products = (
    df.groupby("StockCode")
    .agg(
        product_name=("Description", "first"),
        average_price=("UnitPrice", "mean"),
        total_quantity=("Quantity", "sum"),
        total_sales=("TotalPrice", "sum")
    )
    .reset_index()
)

products["product_name"] = products["product_name"].fillna("Unknown Product")

def assign_category(name):
    name = str(name).upper()

    if any(word in name for word in ["LAMP", "LIGHT", "CANDLE", "LANTERN"]):
        return "Lighting"

    if any(word in name for word in ["CUP", "MUG", "PLATE", "BOWL", "GLASS"]):
        return "Kitchen & Dining"

    if any(word in name for word in ["BAG", "PURSE", "WALLET"]):
        return "Bags & Accessories"

    if any(word in name for word in ["TOY", "GAME", "DOLL"]):
        return "Toys & Games"

    if any(word in name for word in ["HEART", "CHRISTMAS", "EASTER", "DECOR"]):
        return "Home Decor"

    if any(word in name for word in ["BOX", "BASKET", "STORAGE"]):
        return "Storage"

    if any(word in name for word in ["SCARF", "HAT", "CLOTHING", "DRESS"]):
        return "Fashion"

    return "Other"

products["category"] = products["product_name"].apply(assign_category)

products.to_csv(OUTPUT_FILE, index=False)

print("Product feature dataset created")
print(f"Products: {len(products)}")
print(f"Categories: {products['category'].nunique()}")
print("\nCategory distribution:")
print(products["category"].value_counts())
print(f"\nSaved to: {OUTPUT_FILE}")