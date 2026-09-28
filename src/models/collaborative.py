import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "data/processed/user_item_interactions.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    df["CustomerID"] = df["CustomerID"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    return df


def build_item_similarity_model(df):
    user_item = df.pivot_table(
        index="CustomerID",
        columns="StockCode",
        values="interaction",
        aggfunc="sum",
        fill_value=0
    )

    item_user_matrix = user_item.T

    similarity_matrix = cosine_similarity(item_user_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

    return user_item, similarity_df


def recommend_for_customer(
    customer_id,
    user_item,
    similarity_df,
    n=10
):
    customer_id = str(customer_id)

    if customer_id not in user_item.index:
        return pd.DataFrame()

    customer_items = user_item.loc[customer_id]

    purchased_items = customer_items[
        customer_items > 0
    ].index.tolist()

    scores = pd.Series(
        0.0,
        index=similarity_df.index
    )

    for item in purchased_items:
        similar_items = similarity_df[item]

        scores = scores.add(
            similar_items * customer_items[item],
            fill_value=0
        )

    scores = scores.drop(
        labels=purchased_items,
        errors="ignore"
    )

    recommendations = (
        scores
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    recommendations.columns = [
        "StockCode",
        "recommendation_score"
    ]

    return recommendations


if __name__ == "__main__":
    df = load_data()

    print("Building collaborative filtering model...")

    user_item, similarity_df = build_item_similarity_model(df)

    print(f"Users: {user_item.shape[0]}")
    print(f"Products: {user_item.shape[1]}")

    customer_id = user_item.index[0]

    recommendations = recommend_for_customer(
        customer_id,
        user_item,
        similarity_df,
        n=10
    )

    print("\nCollaborative Filtering Recommendations")
    print("=" * 70)

    print(f"Customer ID: {customer_id}")
    print()

    print(recommendations.to_string(index=False))