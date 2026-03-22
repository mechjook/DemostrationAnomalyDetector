"""
Detector de Anomalías — Múltiples métodos estadísticos y ML.

Implementa tres estrategias de detección independientes y las combina
en un score unificado para cada transacción:

  1. Z-Score    — Distancia estadística del monto respecto a la media
  2. IQR        — Rango intercuartílico para outliers robustos
  3. Isolation Forest — Algoritmo de ML para detección no supervisada

Cada método genera un flag binario y un score normalizado [0, 1].
El score final es el promedio ponderado de los tres métodos.

Autor: José Nicolás Candia (@mechjook)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_zscore(df: pd.DataFrame, column: str = "monto",
                  threshold: float = 3.0) -> pd.DataFrame:
    """
    Detección por Z-Score.

    Marca como anomalía toda transacción cuyo monto esté a más de
    `threshold` desviaciones estándar de la media.
    """
    print("\n  [Z-Score] Calculando distancias estadísticas...")
    values = df[column].values.astype(float)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        df["zscore_value"] = 0.0
        df["zscore_flag"] = False
        df["zscore_score"] = 0.0
        return df

    z_scores = np.abs((values - mean) / std)
    df["zscore_value"] = z_scores
    df["zscore_flag"] = z_scores > threshold
    # Normalizar score a [0, 1]
    max_z = z_scores.max()
    df["zscore_score"] = z_scores / max_z if max_z > 0 else 0.0

    flagged = df["zscore_flag"].sum()
    print(f"    Media: ${mean:,.0f} | Std: ${std:,.0f} | Umbral: {threshold}σ")
    print(f"    Flaggeadas: {flagged:,} transacciones ({flagged/len(df)*100:.1f}%)")
    return df


def detect_iqr(df: pd.DataFrame, column: str = "monto",
               k: float = 1.5) -> pd.DataFrame:
    """
    Detección por IQR (Rango Intercuartílico).

    Usa el método de Tukey: outlier si valor < Q1 - k*IQR o > Q3 + k*IQR.
    Más robusto que Z-Score frente a distribuciones sesgadas.
    """
    print("\n  [IQR] Calculando rango intercuartílico...")
    values = df[column].values.astype(float)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    df["iqr_flag"] = (values < lower_bound) | (values > upper_bound)

    # Score basado en distancia al límite más cercano
    distances = np.zeros(len(values))
    below = values < lower_bound
    above = values > upper_bound
    distances[below] = (lower_bound - values[below]) / iqr if iqr > 0 else 0
    distances[above] = (values[above] - upper_bound) / iqr if iqr > 0 else 0
    max_dist = distances.max()
    df["iqr_score"] = distances / max_dist if max_dist > 0 else 0.0

    flagged = df["iqr_flag"].sum()
    print(f"    Q1: ${q1:,.0f} | Q3: ${q3:,.0f} | IQR: ${iqr:,.0f}")
    print(f"    Límites: [${lower_bound:,.0f}, ${upper_bound:,.0f}]")
    print(f"    Flaggeadas: {flagged:,} transacciones ({flagged/len(df)*100:.1f}%)")
    return df


def detect_isolation_forest(df: pd.DataFrame,
                            contamination: float = 0.06,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Detección con Isolation Forest (scikit-learn).

    Usa múltiples features para detectar anomalías multivariantes:
    - Monto absoluto
    - Hora del día
    - Día de la semana (numérico)
    - Frecuencia del destinatario (raro = sospechoso)
    """
    print("\n  [Isolation Forest] Entrenando modelo ML...")

    # Preparar features
    features = pd.DataFrame()
    features["monto_abs"] = df["monto"].abs()
    features["hora"] = pd.to_datetime(df["hora"], format="%H:%M:%S").dt.hour
    features["dia_semana_num"] = pd.to_datetime(df["fecha"]).dt.dayofweek

    # Frecuencia del destinatario (menor frecuencia = más riesgo)
    dest_freq = df["destinatario_rut"].value_counts()
    features["dest_frecuencia"] = df["destinatario_rut"].map(dest_freq)

    # Monto relativo a la categoría
    cat_means = df.groupby("categoria")["monto"].transform("mean")
    cat_stds = df.groupby("categoria")["monto"].transform("std").replace(0, 1)
    features["monto_vs_categoria"] = ((df["monto"] - cat_means) / cat_stds).abs()

    # Es horario nocturno (0-5)
    features["es_nocturno"] = (features["hora"] < 6).astype(int)

    # Es fin de semana
    features["es_finde"] = (features["dia_semana_num"] >= 5).astype(int)

    # Escalar
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # Entrenar Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        max_samples="auto",
        n_jobs=-1,
    )
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    # -1 = anomalía en sklearn, convertir a flag boolean
    df["iforest_flag"] = predictions == -1

    # Normalizar score: decision_function más negativo = más anómalo
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score
    if score_range > 0:
        # Invertir: más negativo → score más alto
        df["iforest_score"] = 1.0 - (scores - min_score) / score_range
    else:
        df["iforest_score"] = 0.0

    flagged = df["iforest_flag"].sum()
    print(f"    Features: {features.shape[1]} | Estimadores: 200")
    print(f"    Contaminación esperada: {contamination*100:.1f}%")
    print(f"    Flaggeadas: {flagged:,} transacciones ({flagged/len(df)*100:.1f}%)")
    return df


def combine_scores(df: pd.DataFrame,
                   weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Combina los scores de los tres métodos en un score final unificado.

    Weights por defecto: Z-Score 0.25, IQR 0.25, Isolation Forest 0.50
    (más peso al método ML que considera múltiples features).
    """
    if weights is None:
        weights = {"zscore": 0.25, "iqr": 0.25, "iforest": 0.50}

    print("\n  [Combinación] Calculando score unificado...")
    print(f"    Pesos: Z-Score={weights['zscore']}, IQR={weights['iqr']}, "
          f"IForest={weights['iforest']}")

    df["score_final"] = (
        df["zscore_score"] * weights["zscore"]
        + df["iqr_score"] * weights["iqr"]
        + df["iforest_score"] * weights["iforest"]
    )

    # Flag combinado: al menos 2 de 3 métodos lo detectan
    df["flag_count"] = (
        df["zscore_flag"].astype(int)
        + df["iqr_flag"].astype(int)
        + df["iforest_flag"].astype(int)
    )
    df["es_anomalia_detectada"] = df["flag_count"] >= 2

    detected = df["es_anomalia_detectada"].sum()
    print(f"    Detectadas (≥2 métodos): {detected:,} transacciones")
    return df


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el pipeline completo de detección de anomalías."""
    print("\n" + "=" * 60)
    print("ETAPA 2: DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)
    print(f"  Analizando {len(df):,} transacciones con 3 métodos...")

    df = detect_zscore(df)
    df = detect_iqr(df)
    df = detect_isolation_forest(df)
    df = combine_scores(df)

    return df
