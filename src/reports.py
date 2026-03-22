"""
Generador de Reportes — CSV con transacciones flaggeadas y resumen ejecutivo.

Produce tres archivos de salida:
  1. reporte_anomalias.csv   — Todas las transacciones detectadas como anómalas
  2. reporte_completo.csv    — Dataset completo con scores y clasificación
  3. resumen_ejecutivo.csv   — KPIs y métricas consolidadas

Autor: José Nicolás Candia (@mechjook)
"""

import csv
import os

import pandas as pd


def _generate_anomaly_report(df: pd.DataFrame, output_dir: str) -> str:
    """Genera reporte con solo las transacciones flaggeadas."""
    anomalies = df[df["es_anomalia_detectada"]].sort_values(
        ["prioridad", "score_final"], ascending=[False, False]
    )

    cols = [
        "id_transaccion", "fecha", "hora", "cuenta_origen", "cuenta_nombre",
        "destinatario_rut", "destinatario_nombre", "categoria", "monto",
        "sucursal", "canal", "score_final", "nivel_riesgo", "factores_riesgo",
        "zscore_flag", "iqr_flag", "iforest_flag", "flag_count",
    ]

    path = os.path.join(output_dir, "reporte_anomalias.csv")
    anomalies[cols].to_csv(path, index=False, encoding="utf-8")
    print(f"    Reporte anomalías : {path} ({len(anomalies):,} registros)")
    return path


def _generate_full_report(df: pd.DataFrame, output_dir: str) -> str:
    """Genera reporte completo con todas las transacciones y sus scores."""
    sorted_df = df.sort_values(["prioridad", "score_final"], ascending=[False, False])

    cols = [
        "id_transaccion", "fecha", "hora", "dia_semana", "cuenta_origen",
        "cuenta_nombre", "destinatario_rut", "destinatario_nombre", "categoria",
        "monto", "sucursal", "canal", "descripcion",
        "score_final", "nivel_riesgo", "factores_riesgo",
        "zscore_value", "zscore_flag", "iqr_flag", "iforest_flag",
        "flag_count", "es_anomalia_detectada",
    ]

    path = os.path.join(output_dir, "reporte_completo.csv")
    sorted_df[cols].to_csv(path, index=False, encoding="utf-8")
    print(f"    Reporte completo  : {path} ({len(sorted_df):,} registros)")
    return path


def _generate_executive_summary(df: pd.DataFrame, output_dir: str) -> str:
    """Genera resumen ejecutivo con métricas consolidadas."""
    total = len(df)
    detected = df["es_anomalia_detectada"].sum()
    monto_total = df["monto"].sum()
    monto_anomalo = df[df["es_anomalia_detectada"]]["monto"].sum()

    # Ground truth (si existe)
    has_ground_truth = "es_anomalia" in df.columns
    if has_ground_truth:
        real_anomalies = df["es_anomalia"].sum()
        true_positives = ((df["es_anomalia"]) & (df["es_anomalia_detectada"])).sum()
        false_positives = ((~df["es_anomalia"]) & (df["es_anomalia_detectada"])).sum()
        false_negatives = ((df["es_anomalia"]) & (~df["es_anomalia_detectada"])).sum()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rows = [
        ("Métrica", "Valor"),
        ("Total transacciones", f"{total:,}"),
        ("Anomalías detectadas", f"{detected:,}"),
        ("Tasa de detección", f"{detected/total*100:.1f}%"),
        ("Monto total procesado", f"${monto_total:,.0f}"),
        ("Monto en anomalías", f"${monto_anomalo:,.0f}"),
        ("% monto anómalo", f"{abs(monto_anomalo)/abs(monto_total)*100:.1f}%"),
        ("Riesgo CRITICO", f"{(df['nivel_riesgo']=='CRITICO').sum():,}"),
        ("Riesgo ALTO", f"{(df['nivel_riesgo']=='ALTO').sum():,}"),
        ("Riesgo MEDIO", f"{(df['nivel_riesgo']=='MEDIO').sum():,}"),
        ("Riesgo BAJO", f"{(df['nivel_riesgo']=='BAJO').sum():,}"),
        ("Score promedio", f"{df['score_final'].mean():.4f}"),
        ("Score máximo", f"{df['score_final'].max():.4f}"),
        ("Método Z-Score flags", f"{df['zscore_flag'].sum():,}"),
        ("Método IQR flags", f"{df['iqr_flag'].sum():,}"),
        ("Método IForest flags", f"{df['iforest_flag'].sum():,}"),
    ]

    if has_ground_truth:
        rows.extend([
            ("--- Validación ---", "---"),
            ("Anomalías reales (ground truth)", f"{real_anomalies:,}"),
            ("True Positives", f"{true_positives:,}"),
            ("False Positives", f"{false_positives:,}"),
            ("False Negatives", f"{false_negatives:,}"),
            ("Precision", f"{precision:.4f}"),
            ("Recall", f"{recall:.4f}"),
            ("F1-Score", f"{f1:.4f}"),
        ])

    path = os.path.join(output_dir, "resumen_ejecutivo.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"    Resumen ejecutivo : {path}")
    return path


def generate_reports(df: pd.DataFrame, output_dir: str) -> dict[str, str]:
    """
    Etapa 4: Genera todos los reportes CSV.

    Retorna diccionario con las rutas de los archivos generados.
    """
    print("\n" + "=" * 60)
    print("ETAPA 4: GENERACIÓN DE REPORTES")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    print(f"  Directorio de salida: {output_dir}")

    paths = {
        "anomalias": _generate_anomaly_report(df, output_dir),
        "completo": _generate_full_report(df, output_dir),
        "resumen": _generate_executive_summary(df, output_dir),
    }

    return paths
