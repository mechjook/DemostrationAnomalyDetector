"""
Clasificador de Riesgo — Scoring de transacciones.

Asigna un nivel de riesgo a cada transacción basándose en el score
combinado de los detectores y reglas de negocio adicionales.

Niveles:
  - BAJO     : score < 0.25 — Transacción normal
  - MEDIO    : 0.25 ≤ score < 0.50 — Requiere revisión
  - ALTO     : 0.50 ≤ score < 0.75 — Sospechosa, escalar
  - CRITICO  : score ≥ 0.75 — Bloqueo inmediato recomendado

Autor: José Nicolás Candia (@mechjook)
"""

import pandas as pd


RISK_THRESHOLDS = {
    "BAJO": (0.0, 0.25),
    "MEDIO": (0.25, 0.50),
    "ALTO": (0.50, 0.75),
    "CRITICO": (0.75, 1.01),
}

RISK_COLORS = {
    "BAJO": "#22C55E",      # verde
    "MEDIO": "#EAB308",     # amarillo
    "ALTO": "#F97316",      # naranja
    "CRITICO": "#EF4444",   # rojo
}


def _assign_risk_level(score: float) -> str:
    """Asigna nivel de riesgo según el score final."""
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "BAJO"


def _compute_risk_factors(row: pd.Series) -> str:
    """Genera una descripción legible de los factores de riesgo detectados."""
    factors = []

    if row.get("zscore_flag", False):
        factors.append(f"Z-Score elevado ({row.get('zscore_value', 0):.1f}σ)")
    if row.get("iqr_flag", False):
        factors.append("Fuera de rango IQR")
    if row.get("iforest_flag", False):
        factors.append("Isolation Forest: anómalo")

    # Reglas de negocio adicionales
    monto = row.get("monto", 0)
    if monto < 0:
        factors.append(f"Monto negativo (${monto:,.0f})")
    elif monto > 15_000_000:
        factors.append(f"Monto muy alto (${monto:,.0f})")

    hora = int(str(row.get("hora", "12:00:00")).split(":")[0])
    if hora < 6:
        factors.append(f"Horario nocturno ({row.get('hora', '')})")

    return " | ".join(factors) if factors else "Sin factores de riesgo"


def classify_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 3: Clasifica cada transacción por nivel de riesgo.

    Agrega columnas:
      - nivel_riesgo: BAJO / MEDIO / ALTO / CRITICO
      - color_riesgo: código hex para visualización
      - factores_riesgo: descripción legible de las señales detectadas
      - prioridad: numérico 1-4 para ordenamiento
    """
    print("\n" + "=" * 60)
    print("ETAPA 3: CLASIFICACIÓN DE RIESGO")
    print("=" * 60)

    df["nivel_riesgo"] = df["score_final"].apply(_assign_risk_level)
    df["color_riesgo"] = df["nivel_riesgo"].map(RISK_COLORS)
    df["factores_riesgo"] = df.apply(_compute_risk_factors, axis=1)

    priority_map = {"BAJO": 1, "MEDIO": 2, "ALTO": 3, "CRITICO": 4}
    df["prioridad"] = df["nivel_riesgo"].map(priority_map)

    # Resumen
    print("\n  Distribución de riesgo:")
    for level in ["BAJO", "MEDIO", "ALTO", "CRITICO"]:
        count = (df["nivel_riesgo"] == level).sum()
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {level:<8} : {count:>5,} ({pct:>5.1f}%) {bar}")

    # Estadísticas por nivel
    print("\n  Monto promedio por nivel:")
    for level in ["BAJO", "MEDIO", "ALTO", "CRITICO"]:
        subset = df[df["nivel_riesgo"] == level]["monto"]
        if len(subset) > 0:
            print(f"    {level:<8} : ${subset.mean():>14,.0f}  "
                  f"(min: ${subset.min():>12,.0f} | max: ${subset.max():>12,.0f})")

    return df
