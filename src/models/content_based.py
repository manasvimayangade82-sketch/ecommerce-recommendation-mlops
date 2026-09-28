import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCT_FILE = "data/processed/product_features.csv"
INTERACTION_FILE = "data/processed/user_item_interactions.csv"


def load_data():
    products = pd.read_csv(PRODUCT_FILE)
    interactions = pd.read_csv(INTERACTION_FILE)

    products["StockCode"] = products["StockCode"].astype(str)
    interactions["CustomerID"] = interactions["CustomerID"].astype(str)
    interactions["StockCode"] = interactions["StockCode"].astype(str)

    products["product_name"] = products["product_name"].fillna("")
    products["category"] = products["category"].fillna("")

    return products, interactions


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

    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=products["StockCode"],
        columns=products["StockCode"]
    )

    return similarity_df


def recommend_for_customer(
    customer_id,
    interactions,
    products,
    similarity_df,
    n=10
):
    customer_id = str(customer_id)

    customer_products = interactions[
        interactions["CustomerID"] == customer_id
    ]["StockCode"].unique()

    if len(customer_products) == 0:
        return products.head(n)[
            ["StockCode", "product_name", "category"]
        ]

    scores = pd.Series(
        0.0,
        index=similarity_df.index
    )

    for product in customer_products:
        if product in similarity_df.index:
            scores = scores.add(
                similarity_df[product],
                fill_value=0
            )

    scores = scores.drop(
        labels=customer_products,
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
            "recommendation_score"
        ]
    ]


if __name__ == "__main__":
    products, interactions = load_data()

    print("Building content-based recommendation model...")

    similarity_df = build_content_model(products)

    print(f"Products: {len(products)}")
    print(f"TF-IDF features created successfully")

    customer_id = interactions["CustomerID"].iloc[0]

    recommendations = recommend_for_customer(
        customer_id,
        interactions,
        products,
        similarity_df,
        n=10
    )

    print("\nContent-Based Recommendations")
    print("=" * 90)

    print(f"Customer ID: {customer_id}")
    print()

    print(
        recommendations.to_string(index=False)
    )