import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

TRANSACTION_FILE = "data/processed/clean_transactions.csv"
PRODUCT_FILE = "data/processed/product_features.csv"

OUTPUT_FILE = "data/processed/model_evaluation.csv"

K = 10
TEST_RATIO = 0.2


def load_data():
    transactions = pd.read_csv(TRANSACTION_FILE)
    products = pd.read_csv(PRODUCT_FILE)

    transactions["CustomerID"] = transactions["CustomerID"].astype(str)
    transactions["StockCode"] = transactions["StockCode"].astype(str)

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        errors="coerce"
    )

    products["StockCode"] = products["StockCode"].astype(str)
    products["product_name"] = products["product_name"].fillna("")
    products["category"] = products["category"].fillna("")

    return transactions, products


def create_time_split(df):
    df = df.sort_values("InvoiceDate").copy()

    train_parts = []
    test_parts = []

    for customer_id, customer_data in df.groupby("CustomerID"):

        customer_data = customer_data.sort_values(
            "InvoiceDate"
        )

        if len(customer_data) < 2:
            train_parts.append(customer_data)
            continue

        split_index = int(
            len(customer_data) * (1 - TEST_RATIO)
        )

        split_index = max(
            1,
            min(split_index, len(customer_data) - 1)
        )

        train_parts.append(
            customer_data.iloc[:split_index]
        )

        test_parts.append(
            customer_data.iloc[split_index:]
        )

    train = pd.concat(
        train_parts,
        ignore_index=True
    )

    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    return train, test


def build_interactions(df):

    return (
        df.groupby(
            ["CustomerID", "StockCode"],
            as_index=False
        )
        .agg(
            interaction=("Quantity", "sum"),
            purchase_count=("InvoiceNo", "nunique"),
            total_spent=("TotalPrice", "sum")
        )
    )


def build_popularity_model(train):

    popularity = (
        train.groupby("StockCode")
        .agg(
            total_interactions=("Quantity", "sum"),
            purchase_count=("InvoiceNo", "nunique"),
            total_spent=("TotalPrice", "sum")
        )
        .reset_index()
    )

    popularity["popularity_score"] = (
        popularity["total_interactions"]
        + popularity["purchase_count"] * 2
        + popularity["total_spent"]
        / max(
            popularity["total_spent"].max(),
            1
        )
    )

    return popularity.sort_values(
        "popularity_score",
        ascending=False
    ).reset_index(drop=True)


def build_collaborative_model(train_interactions):

    user_item = train_interactions.pivot_table(
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

    text_features = (
        products["product_name"].astype(str)
        + " "
        + products["category"].astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        text_features
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

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=scores.index
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


def popularity_recommendations(
    customer_id,
    train_interactions,
    popularity,
    k
):

    purchased = set(
        train_interactions.loc[
            train_interactions["CustomerID"] == customer_id,
            "StockCode"
        ]
    )

    recommendations = popularity[
        ~popularity["StockCode"].isin(purchased)
    ]

    return recommendations[
        "StockCode"
    ].head(k).tolist()


def collaborative_recommendations(
    customer_id,
    user_item,
    similarity,
    k
):

    if customer_id not in user_item.index:
        return []

    customer_items = user_item.loc[
        customer_id
    ]

    purchased = customer_items[
        customer_items > 0
    ]

    if purchased.empty:
        return []

    scores = pd.Series(
        0.0,
        index=similarity.index
    )

    for item, value in purchased.items():

        if item in similarity.index:

            scores = scores.add(
                similarity[item] * value,
                fill_value=0
            )

    scores = scores.drop(
        labels=purchased.index,
        errors="ignore"
    )

    return (
        scores
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist()
    )


def content_recommendations(
    customer_id,
    train_interactions,
    similarity,
    k
):

    purchased = train_interactions.loc[
        train_interactions["CustomerID"] == customer_id,
        "StockCode"
    ].unique()

    purchased = [
        item
        for item in purchased
        if item in similarity.index
    ]

    if len(purchased) == 0:
        return []

    scores = pd.Series(
        0.0,
        index=similarity.index
    )

    for item in purchased:

        scores = scores.add(
            similarity[item],
            fill_value=0
        )

    scores = scores.drop(
        labels=purchased,
        errors="ignore"
    )

    return (
        scores
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist()
    )


def hybrid_recommendations(
    customer_id,
    user_item,
    collaborative_similarity,
    content_similarity,
    k,
    collaborative_weight=0.6,
    content_weight=0.4
):

    if customer_id not in user_item.index:
        return []

    customer_items = user_item.loc[
        customer_id
    ]

    purchased = customer_items[
        customer_items > 0
    ]

    if purchased.empty:
        return []

    collaborative_scores = pd.Series(
        0.0,
        index=collaborative_similarity.index
    )

    content_scores = pd.Series(
        0.0,
        index=content_similarity.index
    )

    for item, value in purchased.items():

        if item in collaborative_similarity.index:

            collaborative_scores = (
                collaborative_scores.add(
                    collaborative_similarity[item]
                    * value,
                    fill_value=0
                )
            )

        if item in content_similarity.index:

            content_scores = (
                content_scores.add(
                    content_similarity[item],
                    fill_value=0
                )
            )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    content_scores = normalize_scores(
        content_scores
    )

    common_products = (
        collaborative_scores.index
        .intersection(
            content_scores.index
        )
    )

    hybrid_scores = (
        collaborative_scores.loc[
            common_products
        ]
        * collaborative_weight
        +
        content_scores.loc[
            common_products
        ]
        * content_weight
    )

    hybrid_scores = hybrid_scores.drop(
        labels=purchased.index,
        errors="ignore"
    )

    return (
        hybrid_scores
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist()
    )


def calculate_metrics(
    recommendations,
    actual_items,
    catalog
):

    recommendations = list(recommendations)
    actual_items = set(actual_items)

    if not recommendations:

        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "hit_rate": 0,
            "ndcg": 0
        }

    hits = [
        1 if item in actual_items else 0
        for item in recommendations
    ]

    hit_count = sum(hits)

    precision = (
        hit_count / len(recommendations)
    )

    recall = (
        hit_count / len(actual_items)
        if actual_items
        else 0
    )

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    hit_rate = (
        1 if hit_count > 0 else 0
    )

    dcg = 0

    for rank, hit in enumerate(
        hits,
        start=1
    ):

        if hit:
            dcg += (
                1 / np.log2(rank + 1)
            )

    ideal_hits = min(
        len(actual_items),
        len(recommendations)
    )

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_rate": hit_rate,
        "ndcg": ndcg
    }


def evaluate_model(
    model_name,
    test,
    catalog,
    recommendation_function,
    model_objects
):

    customers = test[
        "CustomerID"
    ].unique()

    precision_values = []
    recall_values = []
    f1_values = []
    hit_values = []
    ndcg_values = []

    all_recommendations = []

    inference_start = time.time()

    for customer_id in customers:

        actual_items = test.loc[
            test["CustomerID"] == customer_id,
            "StockCode"
        ].unique()

        recommendations = (
            recommendation_function(
                customer_id,
                *model_objects,
                K
            )
        )

        metrics = calculate_metrics(
            recommendations,
            actual_items,
            catalog
        )

        precision_values.append(
            metrics["precision"]
        )

        recall_values.append(
            metrics["recall"]
        )

        f1_values.append(
            metrics["f1"]
        )

        hit_values.append(
            metrics["hit_rate"]
        )

        ndcg_values.append(
            metrics["ndcg"]
        )

        all_recommendations.extend(
            recommendations
        )

    inference_time = (
        time.time()
        - inference_start
    )

    unique_recommendations = set(
        all_recommendations
    )

    coverage = (
        len(unique_recommendations)
        / len(catalog)
    )

    return {
        "model": model_name,
        "precision_at_10": np.mean(
            precision_values
        ),
        "recall_at_10": np.mean(
            recall_values
        ),
        "f1_at_10": np.mean(
            f1_values
        ),
        "hit_rate_at_10": np.mean(
            hit_values
        ),
        "ndcg_at_10": np.mean(
            ndcg_values
        ),
        "catalog_coverage": coverage,
        "inference_time_seconds": (
            inference_time
        )
    }


def main():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    print("=" * 80)
    print("RECOMMENDATION MODEL EVALUATION")
    print("=" * 80)

    transactions, products = load_data()

    print("\nCreating time-based train/test split...")

    train, test = create_time_split(
        transactions
    )

    print(
        f"Training transactions: "
        f"{len(train)}"
    )

    print(
        f"Testing transactions: "
        f"{len(test)}"
    )

    print(
        f"Training customers: "
        f"{train['CustomerID'].nunique()}"
    )

    print(
        f"Testing customers: "
        f"{test['CustomerID'].nunique()}"
    )

    train_interactions = (
        build_interactions(train)
    )

    catalog = set(
        train_interactions[
            "StockCode"
        ].unique()
    )

    print(
        f"Training products: "
        f"{len(catalog)}"
    )

    print("\nBuilding Popularity model...")

    start = time.time()

    popularity = build_popularity_model(
        train
    )

    popularity_training_time = (
        time.time() - start
    )

    print("\nBuilding Collaborative model...")

    start = time.time()

    user_item, collaborative_similarity = (
        build_collaborative_model(
            train_interactions
        )
    )

    collaborative_training_time = (
        time.time() - start
    )

    print("\nBuilding Content-Based model...")

    start = time.time()

    content_similarity = build_content_model(
        products
    )

    content_training_time = (
        time.time() - start
    )

    print("\nEvaluating Popularity...")

    popularity_results = evaluate_model(
        "Popularity",
        test,
        catalog,
        popularity_recommendations,
        [
            train_interactions,
            popularity
        ]
    )

    popularity_results[
        "training_time_seconds"
    ] = popularity_training_time

    print("\nEvaluating Collaborative Filtering...")

    collaborative_results = evaluate_model(
        "Collaborative Filtering",
        test,
        catalog,
        collaborative_recommendations,
        [
            user_item,
            collaborative_similarity
        ]
    )

    collaborative_results[
        "training_time_seconds"
    ] = collaborative_training_time

    print("\nEvaluating Content-Based...")

    content_results = evaluate_model(
        "Content-Based",
        test,
        catalog,
        content_recommendations,
        [
            train_interactions,
            content_similarity
        ]
    )

    content_results[
        "training_time_seconds"
    ] = content_training_time

    print("\nEvaluating Hybrid...")

    start = time.time()

    hybrid_results = evaluate_model(
        "Hybrid",
        test,
        catalog,
        hybrid_recommendations,
        [
            user_item,
            collaborative_similarity,
            content_similarity
        ]
    )

    hybrid_training_time = (
        time.time() - start
    )

    hybrid_results[
        "training_time_seconds"
    ] = (
        collaborative_training_time
        + content_training_time
    )

    results = pd.DataFrame([
        popularity_results,
        collaborative_results,
        content_results,
        hybrid_results
    ])

    results = results[
        [
            "model",
            "precision_at_10",
            "recall_at_10",
            "f1_at_10",
            "hit_rate_at_10",
            "ndcg_at_10",
            "catalog_coverage",
            "training_time_seconds",
            "inference_time_seconds"
        ]
    ]

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved evaluation results to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()