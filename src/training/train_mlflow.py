import os
import time
import joblib
import mlflow
import pandas as pd

from src.evaluation.evaluate_models import (
    load_data,
    create_time_split,
    build_interactions,
    build_popularity_model,
    build_collaborative_model,
    build_content_model
)

MLFLOW_EXPERIMENT = "E-Commerce Recommendation System"
RESULT_FILE = "data/processed/model_evaluation.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_metrics(row):
    mlflow.log_metric("precision_at_10", float(row["precision_at_10"]))
    mlflow.log_metric("recall_at_10", float(row["recall_at_10"]))
    mlflow.log_metric("f1_at_10", float(row["f1_at_10"]))
    mlflow.log_metric("hit_rate_at_10", float(row["hit_rate_at_10"]))
    mlflow.log_metric("ndcg_at_10", float(row["ndcg_at_10"]))
    mlflow.log_metric("catalog_coverage", float(row["catalog_coverage"]))
    mlflow.log_metric("training_time_seconds", float(row["training_time_seconds"]))
    mlflow.log_metric("inference_time_seconds", float(row["inference_time_seconds"]))


def main():
    print("=" * 80)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 80)

    if not os.path.exists(RESULT_FILE):
        raise FileNotFoundError(
            f"{RESULT_FILE} not found. Run evaluate_models.py first."
        )

    results = pd.read_csv(RESULT_FILE)

    transactions, products = load_data()
    train, test = create_time_split(transactions)
    train_interactions = build_interactions(train)

    models = [
        "Popularity",
        "Collaborative Filtering",
        "Content-Based",
        "Hybrid"
    ]

    for model_name in models:
        row = results[results["model"] == model_name].iloc[0]

        print(f"\nLogging experiment: {model_name}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("top_k", 10)
            mlflow.log_param("evaluation_strategy", "time_based_split")
            mlflow.log_param("test_ratio", 0.2)

            if model_name == "Popularity":
                start = time.time()
                model = build_popularity_model(train)
                training_time = time.time() - start

                model_path = f"{MODEL_DIR}/popularity_model.pkl"
                joblib.dump(model, model_path)

            elif model_name == "Collaborative Filtering":
                start = time.time()
                user_item, similarity = build_collaborative_model(
                    train_interactions
                )
                training_time = time.time() - start

                model_path = f"{MODEL_DIR}/collaborative_model.pkl"
                joblib.dump(
                    {
                        "user_item": user_item,
                        "similarity": similarity
                    },
                    model_path
                )

            elif model_name == "Content-Based":
                start = time.time()
                similarity = build_content_model(products)
                training_time = time.time() - start

                model_path = f"{MODEL_DIR}/content_model.pkl"
                joblib.dump(similarity, model_path)

            else:
                mlflow.log_param("collaborative_weight", 0.6)
                mlflow.log_param("content_weight", 0.4)

                start = time.time()

                user_item, collaborative_similarity = build_collaborative_model(
                    train_interactions
                )

                content_similarity = build_content_model(products)

                training_time = time.time() - start

                model_path = f"{MODEL_DIR}/hybrid_model.pkl"

                joblib.dump(
                    {
                        "user_item": user_item,
                        "collaborative_similarity": collaborative_similarity,
                        "content_similarity": content_similarity
                    },
                    model_path
                )

            mlflow.log_metric(
                "actual_training_time_seconds",
                training_time
            )

            log_metrics(row)

            mlflow.log_artifact(model_path)

            print(f"Logged successfully: {model_name}")

    print("\nMLflow experiments completed.")


if __name__ == "__main__":
    main()