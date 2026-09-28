import pandas as pd

INPUT_FILE = "data/processed/user_item_interactions.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    df["CustomerID"] = df["CustomerID"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    return df


def build_popularity_model(df):
    popularity = (
        df.groupby("StockCode")
        .agg(
            total_interactions=("interaction", "sum"),
            purchase_count=("purchase_count", "sum"),
            total_spent=("total_spent", "sum")
        )
        .reset_index()
    )

    popularity["popularity_score"] = (
        popularity["total_interactions"]
        + popularity["purchase_count"] * 2
        + popularity["total_spent"] / popularity["total_spent"].max()
    )

    popularity = popularity.sort_values(
        "popularity_score",
        ascending=False
    ).reset_index(drop=True)

    return popularity


def recommend_for_customer(
    customer_id,
    interactions,
    popularity,
    n=10
):
    customer_id = str(customer_id)

    purchased_products = set(
        interactions.loc[
            interactions["CustomerID"] == customer_id,
            "StockCode"
        ]
    )

    recommendations = popularity[
        ~popularity["StockCode"].isin(purchased_products)
    ].head(n)

    return recommendations


if __name__ == "__main__":
    df = load_data()

    popularity = build_popularity_model(df)

    customer_id = df["CustomerID"].iloc[0]

    recommendations = recommend_for_customer(
        customer_id,
        df,
        popularity,
        n=10
    )

    print("\nPopularity-Based Recommendations")
    print("=" * 70)

    print(f"Customer ID: {customer_id}")
    print()

    print(
        recommendations[
            [
                "StockCode",
                "total_interactions",
                "purchase_count",
                "total_spent",
                "popularity_score"
            ]
        ].to_string(index=False)
    )