import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

INTERACTION_FILE = "data/processed/user_item_interactions.csv"
PRODUCT_FILE = "data/processed/product_features.csv"


def load_data():
    interactions = pd.read_csv(INTERACTION_FILE)
    products = pd.read_csv(PRODUCT_FILE)

    interactions["CustomerID"] = interactions["CustomerID"].astype(str)
    interactions["StockCode"] = interactions["StockCode"].astype(str)
    products["StockCode"] = products["StockCode"].astype(str)

    products["product_name"] = products["product_name"].fillna("")
    products["category"] = products["category"].fillna("")

    return interactions, products


def build_collaborative_model(interactions):
    user_item = interactions.pivot_table(
        index="CustomerID",
        columns="StockCode",
        values="interaction",
        aggfunc="sum",
        fill_value=0
    )

    item_user = user_item.T

    similarity = cosine_similarity(item_user)

    similarity_df = pd.DataFrame(
        similarity,
        index=item_user.index,
        columns=item_user.index
    )

    return user_item, similarity_df


def build_content_model(products):
    products["text_features"] = (
        products["product_name"].astype(str)
        + " "
        + products["category"].astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        products["text_features"]
    )

    similarity = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=products["StockCode"],
        columns=products["StockCode"]
    )

    return similarity_df


def normalize_scores(scores):
    if scores.empty:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return pd.Series(
            0.0,
            index=scores.index
        )

    return (scores - min_score) / (
        max_score - min_score
    )


def recommend_for_customer(
    customer_id,
    interactions,
    products,
    user_item,
    collaborative_similarity,
    content_similarity,
    n=10,
    collaborative_weight=0.6,
    content_weight=0.4
):
    customer_id = str(customer_id)

    if customer_id not in user_item.index:
        return pd.DataFrame()

    customer_items = user_item.loc[customer_id]

    purchased_items = customer_items[
        customer_items > 0
    ].index.tolist()

    collaborative_scores = pd.Series(
        0.0,
        index=collaborative_similarity.index
    )

    content_scores = pd.Series(
        0.0,
        index=content_similarity.index
    )

    for item in purchased_items:

        if item in collaborative_similarity.index:
            collaborative_scores = collaborative_scores.add(
                collaborative_similarity[item]
                * customer_items[item],
                fill_value=0
            )

        if item in content_similarity.index:
            content_scores = content_scores.add(
                content_similarity[item],
                fill_value=0
            )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    content_scores = normalize_scores(
        content_scores
    )

    common_products = (
        collaborative_scores.index
        .intersection(content_scores.index)
    )

    hybrid_scores = (
        collaborative_scores.loc[common_products]
        * collaborative_weight
        +
        content_scores.loc[common_products]
        * content_weight
    )

    hybrid_scores = hybrid_scores.drop(
        labels=purchased_items,
        errors="ignore"
    )

    recommendations = (
        hybrid_scores
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    recommendations.columns = [
        "StockCode",
        "hybrid_score"
    ]

    recommendations = recommendations.merge(
        products[
            ["StockCode", "product_name", "category"]
        ],
        on="StockCode",
        how="left"
    )

    return recommendations[
        [
            "StockCode",
            "product_name",
            "category",
            "hybrid_score"
        ]
    ]


if __name__ == "__main__":

    interactions, products = load_data()

    print("Building hybrid recommendation model...")

    user_item, collaborative_similarity = (
        build_collaborative_model(interactions)
    )

    content_similarity = build_content_model(
        products
    )

    print(f"Customers: {user_item.shape[0]}")
    print(f"Products: {user_item.shape[1]}")

    customer_id = interactions["CustomerID"].iloc[0]

    recommendations = recommend_for_customer(
        customer_id,
        interactions,
        products,
        user_item,
        collaborative_similarity,
        content_similarity,
        n=10,
        collaborative_weight=0.6,
        content_weight=0.4
    )

    print("\nHybrid Recommendations")
    print("=" * 90)

    print(f"Customer ID: {customer_id}")
    print("Collaborative Weight: 0.6")
    print("Content Weight: 0.4")
    print()

    print(
        recommendations.to_string(index=False)
    )