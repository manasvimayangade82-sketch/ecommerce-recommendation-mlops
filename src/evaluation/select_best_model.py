import os
import json
import shutil
import pandas as pd

RESULT_FILE = "data/processed/model_evaluation.csv"
MODEL_DIR = "models"
OUTPUT_FILE = "models/best_model.json"

MODEL_FILES = {
    "Popularity": "popularity_model.pkl",
    "Collaborative Filtering": "collaborative_model.pkl",
    "Content-Based": "content_model.pkl",
    "Hybrid": "hybrid_model.pkl"
}


def main():
    print("=" * 80)
    print("AUTOMATIC BEST MODEL SELECTION")
    print("=" * 80)

    if not os.path.exists(RESULT_FILE):
        raise FileNotFoundError(
            f"{RESULT_FILE} not found. Run evaluate_models.py first."
        )

    results = pd.read_csv(RESULT_FILE)

    required_columns = [
        "model",
        "ndcg_at_10",
        "recall_at_10",
        "f1_at_10",
        "catalog_coverage",
        "inference_time_seconds"
    ]

    missing = [
        column for column in required_columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in evaluation results: {missing}"
        )

    ranked = results.sort_values(
        by=[
            "ndcg_at_10",
            "recall_at_10",
            "f1_at_10",
            "catalog_coverage"
        ],
        ascending=False
    ).reset_index(drop=True)

    best = ranked.iloc[0]

    model_name = best["model"]
    model_file = MODEL_FILES[model_name]
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    shutil.copy2(model_path, best_model_path)

    selection = {
        "selected_model": model_name,
        "model_file": model_file,
        "best_model_path": best_model_path,
        "selection_metric": "ndcg_at_10",
        "selection_strategy": [
            "ndcg_at_10",
            "recall_at_10",
            "f1_at_10",
            "catalog_coverage"
        ],
        "metrics": {
            "precision_at_10": float(best["precision_at_10"]),
            "recall_at_10": float(best["recall_at_10"]),
            "f1_at_10": float(best["f1_at_10"]),
            "hit_rate_at_10": float(best["hit_rate_at_10"]),
            "ndcg_at_10": float(best["ndcg_at_10"]),
            "catalog_coverage": float(best["catalog_coverage"]),
            "training_time_seconds": float(
                best["training_time_seconds"]
            ),
            "inference_time_seconds": float(
                best["inference_time_seconds"]
            )
        }
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(selection, file, indent=4)

    print("\nModel ranking:")
    print(
        ranked[
            [
                "model",
                "ndcg_at_10",
                "recall_at_10",
                "f1_at_10",
                "catalog_coverage"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print(f"SELECTED MODEL: {model_name}")
    print("=" * 80)

    print(f"NDCG@10: {best['ndcg_at_10']:.6f}")
    print(f"Recall@10: {best['recall_at_10']:.6f}")
    print(f"F1@10: {best['f1_at_10']:.6f}")
    print(f"Hit Rate@10: {best['hit_rate_at_10']:.6f}")
    print(f"Catalog Coverage: {best['catalog_coverage']:.6f}")

    print(f"\nBest model copied to: {best_model_path}")
    print(f"Selection metadata saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()