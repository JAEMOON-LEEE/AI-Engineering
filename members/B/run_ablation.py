"""공식 validation에서 클래스 가중치와 PageValues 영향을 비교한다."""

from __future__ import annotations

from run_baseline import (
    FIGURE_DIR,
    METRIC_DIR,
    RANDOM_STATE,
    build_pipeline,
    calculate_metrics,
    load_development_splits,
    prepare_directories,
    preprocess_features,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main() -> None:
    prepare_directories()
    raw_train_x, train_y, raw_validation_x, validation_y = load_development_splits()

    configurations = [
        (
            "Logistic Regression",
            "No class weight",
            "All features",
            LogisticRegression(
                max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
            ),
        ),
        (
            "Logistic Regression",
            "Balanced",
            "All features",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Logistic Regression",
            "Balanced",
            "Without PageValues",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            "No class weight",
            "All features",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            "Balanced",
            "All features",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            "Balanced",
            "Without PageValues",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    rows = []
    for model_name, class_weight_label, feature_set, estimator in configurations:
        current_train = raw_train_x.copy()
        current_validation = raw_validation_x.copy()
        if feature_set == "Without PageValues":
            current_train = current_train.drop(columns=["PageValues"])
            current_validation = current_validation.drop(columns=["PageValues"])

        train_x, validation_x = preprocess_features(
            current_train, current_validation
        )
        pipeline = build_pipeline(estimator)
        pipeline.fit(train_x, train_y)
        probabilities = pipeline.predict_proba(validation_x)[:, 1]
        result = calculate_metrics(
            validation_y,
            probabilities,
            model_name=model_name,
            threshold=0.50,
        )
        result["class_weight"] = class_weight_label
        result["feature_set"] = feature_set
        rows.append(result)

    results = pd.DataFrame(rows)
    column_order = [
        "model",
        "class_weight",
        "feature_set",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    results = results[column_order]
    results.to_csv(
        METRIC_DIR / "validation_ablation_metrics.csv", index=False
    )

    plot_data = results.copy()
    plot_data["configuration"] = (
        plot_data["model"]
        + "\n"
        + plot_data["class_weight"]
        + "\n"
        + plot_data["feature_set"]
    )
    plot_data = plot_data.melt(
        id_vars="configuration",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=plot_data,
        x="configuration",
        y="score",
        hue="metric",
        ax=axis,
    )
    axis.set_title("Official validation: class weight and PageValues ablation")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "validation_ablation_comparison.png", dpi=180
    )
    plt.close(figure)

    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
