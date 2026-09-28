import os
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="MLOps Recommendation Service",
    version="1.0.0"
)

MODEL_FILE = "models/best_model.pkl"
MODEL_INFO_FILE = "models/best_model.json"
TRANSACTIONS_FILE = "data/processed/clean_transactions.csv"
PRODUCTS_FILE = "data/processed/product_features.csv"

model = None
model_info = None
transactions = None
products = None


def load_resources():
    global model
    global model_info
    global transactions
    global products

    model = joblib.load(MODEL_FILE)

    with open(MODEL_INFO_FILE, "r") as file:
        model_info = json.load(file)

    transactions = pd.read_csv(TRANSACTIONS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    transactions["CustomerID"] = transactions["CustomerID"].astype(str)
    transactions["StockCode"] = transactions["StockCode"].astype(str)

    products["StockCode"] = products["StockCode"].astype(str)

    if isinstance(model, pd.DataFrame):
        model.index = model.index.astype(str)
        model.columns = model.columns.astype(str)


@app.on_event("startup")
def startup_event():
    load_resources()


@app.get("/")
def root():
    return {
        "service": "E-Commerce Recommendation API",
        "status": "running",
        "model": model_info["selected_model"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_info["selected_model"]
    }


@app.get("/model-info")
def model_info_endpoint():
    return model_info


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, n: int = 10):

    if n < 1 or n > 50:
        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 50"
        )

    customer_data = transactions[
        transactions["CustomerID"] == customer_id
    ]

    if customer_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    purchased_products = set(
        customer_data["StockCode"].astype(str)
    )

    available_products = [
        code for code in products["StockCode"]
        if code not in purchased_products
    ]

    if not available_products:
        raise HTTPException(
            status_code=404,
            detail="No unseen products available"
        )

    scores = pd.Series(
        0.0,
        index=products["StockCode"]
    )

    valid_history = customer_data[
        customer_data["StockCode"].isin(model.index)
    ]

    if valid_history.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer has no products supported by the model"
        )

    for _, row in valid_history.iterrows():

        purchased_code = str(row["StockCode"])
        quantity = float(row["Quantity"])

        similarity_scores = model.loc[purchased_code]

        similarity_scores = similarity_scores.reindex(
            products["StockCode"]
        ).fillna(0)

        scores = scores.add(
            similarity_scores * quantity,
            fill_value=0
        )

    scores = scores[
        scores.index.isin(available_products)
    ]

    top_recommendations = (
        scores
        .sort_values(ascending=False)
        .head(n)
    )

    recommendation_data = products[
        products["StockCode"].isin(top_recommendations.index)
    ].copy()

    recommendation_data["recommendation_score"] = (
        recommendation_data["StockCode"].map(top_recommendations)
    )

    recommendation_data = recommendation_data.sort_values(
        "recommendation_score",
        ascending=False
    )

    results = []

    for _, row in recommendation_data.iterrows():

        results.append({
            "stock_code": row["StockCode"],
            "product_name": row["product_name"],
            "category": row["category"],
            "average_price": round(
                float(row["average_price"]),
                2
            ),
            "recommendation_score": round(
                float(row["recommendation_score"]),
                6
            )
        })

    return {
        "customer_id": customer_id,
        "model": model_info["selected_model"],
        "recommendation_count": len(results),
        "recommendations": results
    }