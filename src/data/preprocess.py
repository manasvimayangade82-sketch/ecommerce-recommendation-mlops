import os
import pandas as pd

INPUT_FILE = "data/raw/online_retail.xlsx"
OUTPUT_FILE = "data/processed/clean_transactions.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_excel(INPUT_FILE)

initial_rows = len(df)

df = df.dropna(subset=["CustomerID", "StockCode", "Description"])

df = df[df["Quantity"] > 0]

df = df[df["UnitPrice"] > 0]

df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["StockCode"] = df["StockCode"].astype(str)
df["Description"] = df["Description"].str.strip()

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df = df.drop_duplicates()

df = df.sort_values("InvoiceDate")

df.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing completed")
print(f"Initial rows: {initial_rows}")
print(f"Final rows: {len(df)}")
print(f"Customers: {df['CustomerID'].nunique()}")
print(f"Products: {df['StockCode'].nunique()}")
print(f"Transactions: {df['InvoiceNo'].nunique()}")
print(f"Total revenue: {df['TotalPrice'].sum():.2f}")
print(f"Saved to: {OUTPUT_FILE}")