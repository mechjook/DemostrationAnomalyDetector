"""
Analytics — Estadísticas descriptivas y métricas del análisis.

Calcula métricas de rendimiento de la detección, estadísticas
de distribución de montos y correlaciones entre variables.

Autor: José Nicolás Candia (@mechjook)
"""

import numpy as np
import pandas as pd


def compute_detection_metrics(df: pd.DataFrame) -> dict:
    """Calcula precision, recall, F1 y métricas de confusión."""
    has_gt = "es_anomalia" in df.columns
    if not has_gt:
        return {}

    real = df["es_anomalia"]
    pred = df["es_anomalia_detectada"]

    tp = int((real & pred).sum())
    fp = int((~real & pred).sum())
    fn = int((real & ~pred).sum())
    tn = int((~real & ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(df)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def compute_distribution_stats(df: pd.DataFrame) -> dict:
    """Estadísticas descriptivas de montos y scores."""
    montos = df["monto"].values.astype(float)
    scores = df["score_final"].values

    return {
        "monto_mean": float(np.mean(montos)),
        "monto_median": float(np.median(montos)),
        "monto_std": float(np.std(montos)),
        "monto_min": float(np.min(montos)),
        "monto_max": float(np.max(montos)),
        "monto_q1": float(np.percentile(montos, 25)),
        "monto_q3": float(np.percentile(montos, 75)),
        "monto_iqr": float(np.percentile(montos, 75) - np.percentile(montos, 25)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_std": float(np.std(scores)),
        "score_max": float(np.max(scores)),
    }


def compute_category_stats(df: pd.DataFrame) -> dict:
    """Anomalías por categoría de transacción."""
    result = {}
    for cat in df["categoria"].unique():
        subset = df[df["categoria"] == cat]
        detected = subset["es_anomalia_detectada"].sum()
        result[cat] = {
            "total": len(subset),
            "anomalias": int(detected),
            "tasa": detected / len(subset) if len(subset) > 0 else 0,
            "monto_promedio": float(subset["monto"].mean()),
        }
    return result


def run_analytics(df: pd.DataFrame) -> dict:
    """Ejecuta análisis completo y retorna diccionario con todas las métricas."""
    print("\n" + "=" * 60)
    print("ETAPA ANALYTICS: ESTADÍSTICAS Y MÉTRICAS")
    print("=" * 60)

    detection = compute_detection_metrics(df)
    distribution = compute_distribution_stats(df)
    categories = compute_category_stats(df)

    if detection:
        print(f"\n  Métricas de detección:")
        print(f"    Precision : {detection['precision']:.4f}")
        print(f"    Recall    : {detection['recall']:.4f}")
        print(f"    F1-Score  : {detection['f1_score']:.4f}")
        print(f"    Accuracy  : {detection['accuracy']:.4f}")
        print(f"    TP={detection['true_positives']} | FP={detection['false_positives']} | "
              f"FN={detection['false_negatives']} | TN={detection['true_negatives']}")

    print(f"\n  Distribución de montos:")
    print(f"    Media   : ${distribution['monto_mean']:>14,.0f}")
    print(f"    Mediana : ${distribution['monto_median']:>14,.0f}")
    print(f"    Std Dev : ${distribution['monto_std']:>14,.0f}")
    print(f"    IQR     : ${distribution['monto_iqr']:>14,.0f}")

    print(f"\n  Top 5 categorías por tasa de anomalías:")
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["tasa"], reverse=True)
    for cat, stats in sorted_cats[:5]:
        print(f"    {cat:<25} : {stats['tasa']*100:5.1f}% ({stats['anomalias']}/{stats['total']})")

    return {
        "detection": detection,
        "distribution": distribution,
        "categories": categories,
    }
