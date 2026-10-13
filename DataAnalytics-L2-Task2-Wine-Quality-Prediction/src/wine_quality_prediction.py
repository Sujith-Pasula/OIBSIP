"""Wine Quality Prediction — Oasis Infobyte Level 2 Task 2."""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
REPORTS = OUT / "reports"
DATA_PATH = DATA_DIR / "winequality-red.csv"
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
]
SEED = 42


def ensure_dirs():
    for p in (DATA_DIR, CHARTS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def load_data():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, sep=";")
            if "quality" in df.columns and len(df):
                print(f"[INFO] Using existing dataset: {DATA_PATH}")
                return df
        except Exception:
            pass
    errors = []
    for url in DATA_URLS:
        try:
            print(f"[INFO] Downloading dataset from {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
            DATA_PATH.write_bytes(raw)
            # UCI source uses semicolons; infer separator for fallback.
            df = pd.read_csv(DATA_PATH, sep=None, engine="python")
            if "quality" not in df.columns or len(df) == 0:
                raise ValueError("Downloaded CSV does not contain the expected quality column.")
            print(f"[INFO] Loaded {len(df):,} rows and {len(df.columns)} columns.")
            return df
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"[WARN] Download failed: {exc}")
    raise RuntimeError(
        "Could not download the Wine Quality dataset. Check your internet connection, "
        "or manually download the CSV from data/README.md and save it as "
        f"{DATA_PATH}.\n" + "\n".join(errors)
    )


def main():
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    try:
        df = load_data()
        df.columns = [str(c).strip().replace('"', "") for c in df.columns]
        if "quality" not in df.columns:
            raise ValueError(f"Target 'quality' not found. Columns: {list(df.columns)}")

        original_rows = len(df)
        missing = df.isna().sum().to_dict()
        duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().dropna(subset=["quality"]).copy()
        df["quality"] = pd.to_numeric(df["quality"], errors="coerce")
        df = df.dropna(subset=["quality"])
        df["quality"] = df["quality"].astype(int)

        X = df.drop(columns=["quality"]).apply(pd.to_numeric, errors="coerce")
        y = df["quality"]
        # Handle any feature missing values in the modelling pipeline.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )

        class_counts = y.value_counts().sort_index()
        plt.figure(figsize=(9, 5))
        sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
        plt.title("Wine Quality Class Distribution")
        plt.xlabel("Quality score")
        plt.ylabel("Number of samples")
        plt.tight_layout()
        plt.savefig(CHARTS / "quality_class_distribution.png", dpi=160)
        plt.close()

        plt.figure(figsize=(11, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
        plt.title("Wine Features and Quality: Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(CHARTS / "correlation_heatmap.png", dpi=160)
        plt.close()

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1
            ),
            "SGD Classifier": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(
                    loss="log_loss", max_iter=3000, tol=1e-3,
                    class_weight="balanced", random_state=SEED
                ))
            ]),
            "SVC": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(
                    class_weight="balanced", probability=True,
                    random_state=SEED
                ))
            ]),
        }

        metric_rows = []
        report_text = []
        prediction_columns = {"actual_quality": y_test.to_numpy()}
        fitted = {}

        for name, model in models.items():
            print(f"\n[INFO] Training {name}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fitted[name] = model
            prediction_columns[name.replace(" ", "_").lower()] = pred
            acc = accuracy_score(y_test, pred)
            macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
            macro_precision = precision_score(y_test, pred, average="macro", zero_division=0)
            macro_recall = recall_score(y_test, pred, average="macro", zero_division=0)

            # Multiclass ROC-AUC is computed one-vs-rest only when all required classes exist.
            auc = np.nan
            try:
                proba = model.predict_proba(X_test)
                classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
            except Exception:
                pass

            metric_rows.append({
                "model": name,
                "accuracy": float(acc),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "macro_ovr_roc_auc": float(auc) if np.isfinite(auc) else np.nan,
            })
            report_text.append(f"\n{'='*72}\n{name}\n{'='*72}\n")
            report_text.append(classification_report(y_test, pred, zero_division=0))
            cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
            plt.title(f"{name} — Confusion Matrix")
            plt.xlabel("Predicted quality")
            plt.ylabel("Actual quality")
            plt.tight_layout()
            safe = name.lower().replace(" ", "_")
            plt.savefig(CHARTS / f"confusion_matrix_{safe}.png", dpi=160)
            plt.close()
            print(f"{name}: accuracy={acc:.4f}, macro-F1={macro_f1:.4f}, weighted-F1={weighted_f1:.4f}")

        metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
        metrics.to_csv(OUT / "metrics.csv", index=False)
        pd.DataFrame(prediction_columns).to_csv(OUT / "predictions.csv", index=False)
        (OUT / "classification_reports.txt").write_text("\n".join(report_text), encoding="utf-8")

        rf = fitted["Random Forest"]
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        importance.to_csv(OUT / "feature_importance.csv", index=False)
        plt.figure(figsize=(10, 7))
        sns.barplot(data=importance.head(12).sort_values("importance"),
                    x="importance", y="feature")
        plt.title("Random Forest — Top Feature Importances")
        plt.tight_layout()
        plt.savefig(CHARTS / "feature_importance.png", dpi=160)
        plt.close()

        best = metrics.iloc[0]
        imbalance_ratio = float(class_counts.max() / class_counts.min())
        report_lines = [
            "# Wine Quality Prediction — Run Report", "",
            "## Dataset checks",
            f"- Rows before cleaning: **{original_rows:,}**",
            f"- Rows after duplicate/target cleanup: **{len(df):,}**",
            f"- Columns including target: **{df.shape[1]}**",
            f"- Duplicate rows found before cleanup: **{duplicates}**",
            f"- Missing values by original column: `{json.dumps({str(k): int(v) for k, v in missing.items()})}`",
            f"- Quality classes: **{len(class_counts)}**",
            f"- Majority/minority class count ratio: **{imbalance_ratio:.2f}:1**",
            "", "## Quality class counts", "",
            "| Quality | Samples |", "|---:|---:|"
        ]
        for label, count in class_counts.items():
            report_lines.append(f"| {label} | {count} |")
        report_lines += [
            "", "## Model comparison", "",
            "| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Macro OVR ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|"
        ]
        for row in metrics.to_dict(orient="records"):
            auc_s = f"{row['macro_ovr_roc_auc']:.4f}" if pd.notna(row["macro_ovr_roc_auc"]) else "N/A"
            report_lines.append(
                f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_precision']:.4f} | "
                f"{row['macro_recall']:.4f} | {row['macro_f1']:.4f} | "
                f"{row['weighted_f1']:.4f} | {auc_s} |"
            )
        report_lines += [
            "", f"Best macro-F1 in this run: **{best['model']}** ({best['macro_f1']:.4f}).",
            "This is the result of this train/test split, not a guarantee of general performance.",
            "", "## Interpretation prompts",
            "- Which quality score has the most and fewest samples?",
            "- Does accuracy tell the whole story? Compare it with macro-F1 and per-class recall.",
            "- Which features are most important in the Random Forest model?",
            "- Which model performs best on minority classes according to the classification report?",
            "- Would grouping scores into broader quality bands make sense for a different business objective? This project preserves original scores.",
            "", "## Limitations",
            "- This dataset contains red wine samples from a specific study, not all wine types or markets.",
            "- Quality scores are subjective sensory ratings and classes are imbalanced.",
            "- Model metrics should not be presented as proof of real-world wine certification performance.",
            "- Use the metrics generated by this run in your final README and demo."
        ]
        (REPORTS / "eda_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        (OUT / "summary.json").write_text(json.dumps({
            "rows_before_cleaning": original_rows,
            "rows_after_cleaning": len(df),
            "columns": list(df.columns),
            "quality_class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "stratified_test_size": 0.20,
            "random_state": SEED,
            "metrics": metric_rows,
            "best_by_macro_f1": str(best["model"]),
        }, indent=2), encoding="utf-8")

        print("\n[SUCCESS] Wine Quality Prediction completed.")
        print(f"[INFO] Metrics: {OUT / 'metrics.csv'}")
        print(f"[INFO] Report: {REPORTS / 'eda_report.md'}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
