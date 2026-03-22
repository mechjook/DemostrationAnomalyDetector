"""
Detector de Anomalías Financieras — Pipeline Principal
======================================================
Demostración de capacidades de análisis de datos avanzado y
detección de patrones anómalos en transacciones financieras.

Etapas:
  0. Generación de datos sintéticos (~5000 transacciones)
  1. Validación de estructura e integridad
  2. Detección de anomalías (Z-Score, IQR, Isolation Forest)
  3. Clasificación de riesgo (bajo/medio/alto/crítico)
  4. Generación de reportes CSV
  5. Generación de dashboard HTML interactivo
  A. Analytics — Estadísticas y métricas de rendimiento

Autor: José Nicolás Candia (@mechjook)
"""

import os
import sys
import time

import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def main():
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   DETECTOR DE ANOMALÍAS FINANCIERAS — PIPELINE AUTOMÁTICO   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Etapa 0: Generación de datos ---
    from src.generate_data import generate_transactions
    print("\n" + "=" * 60)
    print("ETAPA 0: GENERACIÓN DE DATOS SINTÉTICOS")
    print("=" * 60)
    data_path = generate_transactions()

    # --- Etapa 1: Validación ---
    from validators.data_validator import validate_all
    result = validate_all(data_path)
    if not result.is_valid:
        print("\n⚠ DATOS DE ENTRADA INVÁLIDOS — PIPELINE DETENIDO")
        sys.exit(1)

    # Cargar dataset
    print("\n  Cargando dataset...")
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"  Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")

    # --- Etapa 2: Detección de anomalías ---
    from src.detector import run_detection
    df = run_detection(df)

    # --- Etapa 3: Clasificación de riesgo ---
    from src.classifier import classify_risk
    df = classify_risk(df)

    # --- Etapa 4: Reportes ---
    from src.reports import generate_reports
    report_paths = generate_reports(df, OUTPUT_DIR)

    # --- Etapa 5: Dashboard ---
    from src.dashboard import generate_dashboard
    dashboard_path = generate_dashboard(df, OUTPUT_DIR)

    # --- Analytics ---
    from src.analytics import run_analytics
    stats = run_analytics(df)

    # --- Resumen final ---
    elapsed = time.time() - start
    detected = df["es_anomalia_detectada"].sum()
    real = df["es_anomalia"].sum()
    precision = stats["detection"].get("precision", 0)
    recall = stats["detection"].get("recall", 0)
    f1 = stats["detection"].get("f1_score", 0)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"  Tiempo total       : {elapsed:.2f}s")
    print(f"  Transacciones      : {len(df):,}")
    print(f"  Anomalías reales   : {real:,}")
    print(f"  Anomalías detectadas: {detected:,}")
    print(f"  Precision          : {precision:.4f}")
    print(f"  Recall             : {recall:.4f}")
    print(f"  F1-Score           : {f1:.4f}")
    print(f"  Dashboard          : {dashboard_path}")
    print(f"  Reportes en        : {OUTPUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
