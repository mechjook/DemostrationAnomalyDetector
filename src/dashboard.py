"""
Dashboard Interactivo — HTML + Chart.js para visualización de anomalías.

Genera una página HTML autocontenida con:
  - KPIs animados (total, detectadas, precision, recall)
  - Scatter plot de anomalías (monto vs score)
  - Distribución de montos (histograma)
  - Timeline de anomalías por mes
  - Distribución por tipo de anomalía (doughnut)
  - Heatmap hora vs día de la semana
  - Tabla de alertas filtrable y paginada

Dark theme responsive con Chart.js 4.

Autor: José Nicolás Candia (@mechjook)
"""

import json
import os

import pandas as pd


def _prepare_chart_data(df: pd.DataFrame) -> dict:
    """Prepara todos los datos que consumirán los gráficos."""
    data = {}

    # --- KPIs ---
    total = len(df)
    detected = int(df["es_anomalia_detectada"].sum())
    has_gt = "es_anomalia" in df.columns
    real = int(df["es_anomalia"].sum()) if has_gt else 0
    tp = int(((df["es_anomalia"]) & (df["es_anomalia_detectada"])).sum()) if has_gt else 0
    fp = int(((~df["es_anomalia"]) & (df["es_anomalia_detectada"])).sum()) if has_gt else 0
    fn = int(((df["es_anomalia"]) & (~df["es_anomalia_detectada"])).sum()) if has_gt else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    monto_total = float(df["monto"].sum())
    monto_anomalo = float(df[df["es_anomalia_detectada"]]["monto"].sum())
    score_avg = float(df["score_final"].mean())

    data["kpis"] = {
        "total": total,
        "detected": detected,
        "real_anomalies": real,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "monto_total": monto_total,
        "monto_anomalo": monto_anomalo,
        "score_avg": round(score_avg, 4),
        "critico": int((df["nivel_riesgo"] == "CRITICO").sum()),
        "alto": int((df["nivel_riesgo"] == "ALTO").sum()),
        "medio": int((df["nivel_riesgo"] == "MEDIO").sum()),
        "bajo": int((df["nivel_riesgo"] == "BAJO").sum()),
    }

    # --- Scatter: monto vs score (muestra de 2000 puntos) ---
    sample_n = min(2000, len(df))
    sample = df.sample(n=sample_n, random_state=42)
    scatter_normal = sample[~sample["es_anomalia_detectada"]]
    scatter_anomaly = sample[sample["es_anomalia_detectada"]]

    data["scatter"] = {
        "normal": {
            "x": scatter_normal["score_final"].round(4).tolist(),
            "y": scatter_normal["monto"].tolist(),
        },
        "anomaly": {
            "x": scatter_anomaly["score_final"].round(4).tolist(),
            "y": scatter_anomaly["monto"].tolist(),
        },
    }

    # --- Histograma de montos ---
    import numpy as np
    monto_values = df["monto"].values
    # Usar percentiles para evitar que outliers extremos rompan el histograma
    p1 = float(np.percentile(monto_values, 1))
    p99 = float(np.percentile(monto_values, 99))
    filtered = monto_values[(monto_values >= p1) & (monto_values <= p99)]
    hist_counts, hist_edges = np.histogram(filtered, bins=30)
    data["histogram"] = {
        "labels": [f"${(hist_edges[i]+hist_edges[i+1])/2:,.0f}" for i in range(len(hist_counts))],
        "counts": hist_counts.tolist(),
        "edges": hist_edges.tolist(),
    }

    # --- Timeline: anomalías por mes ---
    df_copy = df.copy()
    df_copy["mes"] = pd.to_datetime(df_copy["fecha"]).dt.to_period("M").astype(str)
    monthly_total = df_copy.groupby("mes").size()
    monthly_anomalies = df_copy[df_copy["es_anomalia_detectada"]].groupby("mes").size()
    months = sorted(df_copy["mes"].unique())
    data["timeline"] = {
        "labels": months,
        "total": [int(monthly_total.get(m, 0)) for m in months],
        "anomalias": [int(monthly_anomalies.get(m, 0)) for m in months],
    }

    # --- Distribución por tipo de anomalía ---
    if has_gt:
        anomaly_types = df[df["es_anomalia"]]["tipo_anomalia"].value_counts()
        data["anomaly_types"] = {
            "labels": anomaly_types.index.tolist(),
            "counts": anomaly_types.values.tolist(),
        }
    else:
        data["anomaly_types"] = {"labels": [], "counts": []}

    # --- Distribución por nivel de riesgo ---
    risk_dist = df["nivel_riesgo"].value_counts()
    data["risk_distribution"] = {
        "labels": ["BAJO", "MEDIO", "ALTO", "CRITICO"],
        "counts": [int(risk_dist.get(l, 0)) for l in ["BAJO", "MEDIO", "ALTO", "CRITICO"]],
        "colors": ["#22C55E", "#EAB308", "#F97316", "#EF4444"],
    }

    # --- Heatmap: hora vs día de la semana (solo anomalías) ---
    anomalias_df = df[df["es_anomalia_detectada"]].copy()
    anomalias_df["hora_num"] = pd.to_datetime(anomalias_df["hora"], format="%H:%M:%S").dt.hour
    anomalias_df["dia_num"] = pd.to_datetime(anomalias_df["fecha"]).dt.dayofweek
    heatmap = [[0]*7 for _ in range(24)]
    for _, row in anomalias_df.iterrows():
        h = int(row["hora_num"])
        d = int(row["dia_num"])
        heatmap[h][d] += 1
    data["heatmap"] = {
        "data": heatmap,
        "hours": [f"{h:02d}:00" for h in range(24)],
        "days": ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
    }

    # --- Métodos de detección: Venn-like data ---
    data["detection_methods"] = {
        "zscore_only": int(((df["zscore_flag"]) & (~df["iqr_flag"]) & (~df["iforest_flag"])).sum()),
        "iqr_only": int(((~df["zscore_flag"]) & (df["iqr_flag"]) & (~df["iforest_flag"])).sum()),
        "iforest_only": int(((~df["zscore_flag"]) & (~df["iqr_flag"]) & (df["iforest_flag"])).sum()),
        "zscore_iqr": int(((df["zscore_flag"]) & (df["iqr_flag"]) & (~df["iforest_flag"])).sum()),
        "zscore_iforest": int(((df["zscore_flag"]) & (~df["iqr_flag"]) & (df["iforest_flag"])).sum()),
        "iqr_iforest": int(((~df["zscore_flag"]) & (df["iqr_flag"]) & (df["iforest_flag"])).sum()),
        "all_three": int(((df["zscore_flag"]) & (df["iqr_flag"]) & (df["iforest_flag"])).sum()),
        "zscore_total": int(df["zscore_flag"].sum()),
        "iqr_total": int(df["iqr_flag"].sum()),
        "iforest_total": int(df["iforest_flag"].sum()),
    }

    # --- Top 20 transacciones más sospechosas ---
    top = df.nlargest(20, "score_final")
    data["top_alerts"] = []
    for _, row in top.iterrows():
        data["top_alerts"].append({
            "id": row["id_transaccion"],
            "fecha": row["fecha"],
            "hora": row["hora"],
            "destinatario": row["destinatario_nombre"],
            "rut": row["destinatario_rut"],
            "categoria": row["categoria"],
            "monto": float(row["monto"]),
            "score": round(float(row["score_final"]), 4),
            "nivel": row["nivel_riesgo"],
            "factores": row["factores_riesgo"],
            "sucursal": row["sucursal"],
            "canal": row["canal"],
        })

    # --- Tabla completa de anomalías (para la tabla paginada) ---
    all_anomalies = df[df["es_anomalia_detectada"]].sort_values("score_final", ascending=False)
    data["all_anomalies"] = []
    for _, row in all_anomalies.iterrows():
        data["all_anomalies"].append({
            "id": row["id_transaccion"],
            "fecha": row["fecha"],
            "hora": row["hora"],
            "destinatario": row["destinatario_nombre"],
            "rut": row["destinatario_rut"],
            "categoria": row["categoria"],
            "monto": float(row["monto"]),
            "score": round(float(row["score_final"]), 4),
            "nivel": row["nivel_riesgo"],
            "color": row["color_riesgo"],
            "factores": row["factores_riesgo"],
            "sucursal": row["sucursal"],
            "canal": row["canal"],
        })

    # --- Score distribution ---
    scores = df["score_final"].values
    score_hist, score_edges = pd.cut(scores, bins=50, retbins=True)
    score_counts = pd.Series(scores).groupby(pd.cut(scores, bins=50)).count()
    data["score_distribution"] = {
        "labels": [f"{score_edges[i]:.2f}" for i in range(len(score_edges)-1)],
        "counts": score_counts.values.tolist(),
    }

    return data


def generate_dashboard(df: pd.DataFrame, output_dir: str) -> str:
    """
    Etapa 5: Genera el dashboard HTML interactivo.

    Retorna la ruta del archivo index.html generado.
    """
    print("\n" + "=" * 60)
    print("ETAPA 5: GENERACIÓN DE DASHBOARD")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    print("  Preparando datos para gráficos...")
    chart_data = _prepare_chart_data(df)
    chart_data_json = json.dumps(chart_data, ensure_ascii=False)

    print("  Generando HTML interactivo...")
    html = _build_html(chart_data_json)

    output_path = os.path.join(output_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Dashboard generado: {output_path}")
    return output_path


def _build_html(chart_data_json: str) -> str:
    """Construye el HTML completo del dashboard."""
    return f'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Detector de Anomalías Financieras — Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-primary: #0F172A;
      --bg-secondary: #1E293B;
      --bg-card: #1E293B;
      --bg-card-hover: #334155;
      --text-primary: #E2E8F0;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      --accent-blue: #3B82F6;
      --accent-green: #22C55E;
      --accent-yellow: #EAB308;
      --accent-orange: #F97316;
      --accent-red: #EF4444;
      --accent-purple: #A855F7;
      --accent-cyan: #06B6D4;
      --border-color: #334155;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
    }}

    /* --- NAVBAR --- */
    .navbar {{
      position: fixed; top: 0; left: 0; right: 0; z-index: 100;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      padding: 0 2rem;
      display: flex; align-items: center; justify-content: space-between;
      height: 56px;
      backdrop-filter: blur(12px);
    }}
    .navbar-brand {{
      display: flex; align-items: center; gap: 10px;
      font-weight: 700; font-size: 1.1rem;
    }}
    .navbar-brand .icon {{ font-size: 1.4rem; }}
    .nav-links {{
      display: flex; gap: 4px; overflow-x: auto;
    }}
    .nav-links a {{
      color: var(--text-secondary); text-decoration: none; padding: 6px 14px;
      border-radius: 6px; font-size: 0.85rem; white-space: nowrap;
      transition: all 0.2s;
    }}
    .nav-links a:hover, .nav-links a.active {{
      color: var(--text-primary); background: var(--bg-card-hover);
    }}
    .status-badge {{
      display: flex; align-items: center; gap: 6px;
      background: rgba(34,197,94,0.15); color: var(--accent-green);
      padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
    }}
    .status-dot {{
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--accent-green);
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.4; }}
    }}

    /* --- LAYOUT --- */
    .container {{
      max-width: 1400px; margin: 0 auto;
      padding: 76px 1.5rem 3rem;
    }}
    section {{ margin-bottom: 3rem; scroll-margin-top: 70px; }}
    .section-title {{
      font-size: 1.3rem; font-weight: 700;
      margin-bottom: 1.5rem; padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--accent-blue);
      display: flex; align-items: center; gap: 8px;
    }}

    /* --- GRID --- */
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(400px,100%), 1fr));
      gap: 1.5rem;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(320px,100%), 1fr));
      gap: 1.5rem;
    }}

    /* --- KPI CARDS --- */
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px; padding: 1.2rem;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: var(--accent-blue);
    }}
    .kpi-label {{
      font-size: 0.8rem; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    .kpi-value {{
      font-size: 1.8rem; font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .kpi-detail {{
      font-size: 0.78rem; color: var(--text-secondary);
      margin-top: 4px;
    }}

    /* --- CHART CARD --- */
    .chart-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px; padding: 1.5rem;
    }}
    .chart-card h3 {{
      font-size: 1rem; margin-bottom: 1rem;
      color: var(--text-secondary);
    }}
    .chart-card canvas {{
      max-height: 350px;
    }}

    /* --- HEATMAP --- */
    .heatmap-grid {{
      display: grid;
      grid-template-columns: 50px repeat(7, 1fr);
      gap: 2px; font-size: 0.7rem;
    }}
    .heatmap-cell {{
      padding: 4px; text-align: center;
      border-radius: 3px; min-height: 22px;
      transition: transform 0.1s;
    }}
    .heatmap-cell:hover {{ transform: scale(1.3); z-index: 2; }}
    .heatmap-header {{
      color: var(--text-muted); font-weight: 600;
      display: flex; align-items: center; justify-content: center;
    }}

    /* --- TABLE --- */
    .table-wrapper {{
      overflow-x: auto;
    }}
    .data-table {{
      width: 100%; border-collapse: collapse;
      font-size: 0.82rem;
    }}
    .data-table th {{
      background: var(--bg-primary);
      color: var(--text-muted); text-transform: uppercase;
      font-size: 0.72rem; letter-spacing: 0.05em;
      padding: 10px 12px; text-align: left;
      border-bottom: 2px solid var(--border-color);
      cursor: pointer; white-space: nowrap;
      position: sticky; top: 0;
    }}
    .data-table th:hover {{ color: var(--text-primary); }}
    .data-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid var(--border-color);
      white-space: nowrap;
    }}
    .data-table tr:hover td {{
      background: var(--bg-card-hover);
    }}
    .risk-badge {{
      padding: 2px 8px; border-radius: 12px;
      font-size: 0.72rem; font-weight: 600;
    }}

    /* --- SEARCH & FILTERS --- */
    .controls {{
      display: flex; gap: 1rem; margin-bottom: 1rem;
      flex-wrap: wrap; align-items: center;
    }}
    .search-input {{
      background: var(--bg-primary); border: 1px solid var(--border-color);
      border-radius: 8px; padding: 8px 14px;
      color: var(--text-primary); font-size: 0.85rem;
      min-width: min(250px,100%); outline: none;
    }}
    .search-input:focus {{ border-color: var(--accent-blue); }}
    .filter-btn {{
      background: var(--bg-card); border: 1px solid var(--border-color);
      color: var(--text-secondary); padding: 6px 14px;
      border-radius: 8px; cursor: pointer; font-size: 0.82rem;
      transition: all 0.2s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      border-color: var(--accent-blue); color: var(--text-primary);
      background: rgba(59,130,246,0.1);
    }}

    /* --- PAGINATION --- */
    .pagination {{
      display: flex; gap: 4px; justify-content: center;
      margin-top: 1rem; align-items: center;
    }}
    .page-btn {{
      background: var(--bg-card); border: 1px solid var(--border-color);
      color: var(--text-secondary); padding: 6px 12px;
      border-radius: 6px; cursor: pointer; font-size: 0.82rem;
    }}
    .page-btn:hover, .page-btn.active {{
      border-color: var(--accent-blue); color: var(--text-primary);
    }}
    .page-info {{
      color: var(--text-muted); font-size: 0.8rem;
      margin: 0 8px;
    }}

    /* --- METHODS COMPARISON --- */
    .method-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px; padding: 1.2rem;
      text-align: center;
    }}
    .method-name {{
      font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem;
    }}
    .method-count {{
      font-size: 2rem; font-weight: 800;
    }}

    /* --- PROGRESS BAR --- */
    .progress-bg {{
      background: var(--bg-primary); border-radius: 6px;
      height: 8px; overflow: hidden; margin-top: 6px;
    }}
    .progress-fill {{
      height: 100%; border-radius: 6px;
      transition: width 1.5s ease-out;
    }}

    /* --- ANIMATION --- */
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{
      animation: fadeIn 0.5s ease-out;
    }}

    /* --- SCROLLBAR --- */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
    ::-webkit-scrollbar-thumb {{
      background: var(--border-color); border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}

    /* --- FOOTER --- */
    footer {{
      text-align: center; color: var(--text-muted);
      font-size: 0.8rem; padding: 2rem 0;
      border-top: 1px solid var(--border-color);
    }}
    footer a {{ color: var(--accent-blue); text-decoration: none; }}

    /* --- HERO / INTRO --- */
    .hero {{
      background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(168,85,247,0.08));
      border: 1px solid var(--border-color);
      border-radius: 16px; padding: 2rem 2.5rem;
      margin-bottom: 2.5rem;
    }}
    .hero h1 {{
      font-size: 1.6rem; margin-bottom: 0.5rem;
    }}
    .hero .subtitle {{
      color: var(--text-secondary); font-size: 0.95rem;
      margin-bottom: 1.2rem; max-width: 800px;
    }}
    .pipeline-flow {{
      display: flex; flex-wrap: wrap; gap: 0.5rem;
      align-items: center; margin-top: 1rem;
    }}
    .pipeline-step {{
      background: var(--bg-card); border: 1px solid var(--border-color);
      border-radius: 8px; padding: 6px 14px; font-size: 0.8rem;
      color: var(--text-secondary); white-space: nowrap;
    }}
    .pipeline-step strong {{ color: var(--text-primary); }}
    .pipeline-arrow {{
      color: var(--accent-blue); font-size: 1rem; font-weight: 700;
    }}

    /* --- SECTION DESCRIPTION --- */
    .section-desc {{
      color: var(--text-secondary); font-size: 0.88rem;
      margin: -0.8rem 0 1.5rem 0;
      max-width: 900px; line-height: 1.7;
    }}
    .section-desc code {{
      background: rgba(59,130,246,0.12); color: var(--accent-cyan);
      padding: 1px 6px; border-radius: 4px; font-size: 0.82rem;
    }}
    .chart-desc {{
      color: var(--text-muted); font-size: 0.78rem;
      margin-top: -0.5rem; margin-bottom: 0.8rem;
      line-height: 1.5;
    }}

    /* RESPONSIVE */
    @media(max-width:768px) {{
      .navbar {{ padding: 0 1rem; flex-wrap: wrap; height: auto; min-height: 56px; padding-top: 0.5rem; padding-bottom: 0.5rem; }}
      .navbar-brand {{ font-size: 0.95rem; }}
      .nav-links {{ order: 3; width: 100%; padding-bottom: 0.25rem; }}
      .nav-links a {{ padding: 4px 10px; font-size: 0.78rem; }}
      .status-badge {{ display: none; }}
      .container {{ padding: 90px 1rem 2rem; }}
      .hero {{ padding: 1.5rem; }}
      .hero h1 {{ font-size: 1.3rem; line-height: 1.3; }}
      .hero .subtitle {{ font-size: 0.85rem; }}
      .section-title {{ font-size: 1.1rem; }}
      .section-desc {{ font-size: 0.82rem; }}
      .grid-4 {{ grid-template-columns: 1fr 1fr; }}
      .kpi-value {{ font-size: 1.4rem; }}
      .kpi-card {{ padding: 1rem; }}
      .chart-card {{ padding: 1rem; }}
      .chart-card h3 {{ font-size: 0.9rem; }}
      .heatmap-grid {{ font-size: 0.6rem; gap: 1px; }}
      .data-table th, .data-table td {{ padding: 6px 8px; }}
      footer {{ padding: 1.5rem 1rem; }}
    }}
    @media(max-width:480px) {{
      .grid-4 {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 1.1rem; }}
      .kpi-value {{ font-size: 1.2rem; }}
      .heatmap-grid {{ grid-template-columns: 30px repeat(7, 1fr); }}
      .pipeline-step {{ font-size: 0.7rem; padding: 4px 10px; }}
    }}
  </style>
</head>
<body>

<!-- NAVBAR -->
<nav class="navbar">
  <div class="navbar-brand">
    <span class="icon">&#128269;</span>
    <span>Anomaly Detector</span>
  </div>
  <div class="nav-links">
    <a href="#resumen" class="active">Resumen</a>
    <a href="#deteccion">Detección</a>
    <a href="#distribucion">Distribución</a>
    <a href="#timeline">Timeline</a>
    <a href="#heatmap">Heatmap</a>
    <a href="#alertas">Alertas</a>
    <a href="#acerca">Acerca de</a>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    Análisis completo
  </div>
</nav>

<div class="container">

  <!-- HERO INTRO -->
  <div class="hero fade-in">
    <h1>Detector de Anomalías Financieras</h1>
    <p class="subtitle">
      Este sistema analiza un dataset de <strong>~5.000 transacciones financieras sintéticas</strong>
      para identificar operaciones sospechosas. Los datos simulan movimientos de cuentas corrientes
      de una empresa chilena durante 2025, con anomalías inyectadas intencionalmente: montos
      atípicos, transacciones en horarios inusuales, ráfagas de operaciones, duplicados y
      transferencias a cuentas desconocidas.
    </p>
    <p class="subtitle" style="margin-bottom:0.5rem">
      El pipeline aplica <strong>3 métodos de detección</strong> (Z-Score, IQR e Isolation Forest)
      y combina sus resultados para clasificar cada transacción por nivel de riesgo. Este dashboard
      presenta los hallazgos del análisis completo.
    </p>
    <div class="pipeline-flow">
      <div class="pipeline-step"><strong>1.</strong> Generación de datos</div>
      <span class="pipeline-arrow">&#8594;</span>
      <div class="pipeline-step"><strong>2.</strong> Validación</div>
      <span class="pipeline-arrow">&#8594;</span>
      <div class="pipeline-step"><strong>3.</strong> Detección (Z-Score + IQR + IForest)</div>
      <span class="pipeline-arrow">&#8594;</span>
      <div class="pipeline-step"><strong>4.</strong> Clasificación de riesgo</div>
      <span class="pipeline-arrow">&#8594;</span>
      <div class="pipeline-step"><strong>5.</strong> Reportes CSV</div>
      <span class="pipeline-arrow">&#8594;</span>
      <div class="pipeline-step"><strong>6.</strong> Dashboard</div>
    </div>
  </div>

  <!-- RESUMEN KPIs -->
  <section id="resumen" class="fade-in">
    <div class="section-title">&#128202; Resumen Ejecutivo</div>
    <p class="section-desc">
      Indicadores clave del análisis. <strong>Precision</strong> mide qué porcentaje de las
      anomalías detectadas son realmente anómalas (evita falsas alarmas).
      <strong>Recall</strong> mide qué porcentaje de las anomalías reales fueron encontradas
      (evita que se escapen). <strong>F1-Score</strong> es el balance entre ambas.
    </p>
    <div class="grid-4" id="kpi-grid"></div>
  </section>

  <!-- DETECCIÓN -->
  <section id="deteccion" class="fade-in">
    <div class="section-title">&#128300; Análisis de Detección</div>
    <p class="section-desc">
      Cada transacción recibe un <strong>score de 0 a 1</strong> combinando tres métodos:
      <code>Z-Score</code> (distancia estadística del monto respecto a la media),
      <code>IQR</code> (rango intercuartílico, robusto ante distribuciones sesgadas) e
      <code>Isolation Forest</code> (algoritmo de ML que considera 7 variables: monto, hora,
      día, frecuencia del destinatario, etc.). Una transacción se marca como anómala si
      <strong>al menos 2 de los 3 métodos</strong> la detectan.
    </p>
    <div class="grid-2">
      <div class="chart-card">
        <h3>Scatter: Score vs Monto</h3>
        <p class="chart-desc">
          Cada punto es una transacción. El eje X es el score de sospecha (más alto = más anómalo)
          y el eje Y es el monto. Los puntos rojos son anomalías detectadas; los azules, transacciones
          normales. Las anomalías tienden a concentrarse en la esquina superior derecha (alto monto + alto score).
        </p>
        <canvas id="scatterChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Distribución de Scores</h3>
        <p class="chart-desc">
          Histograma de los scores de todas las transacciones. La gran mayoría se concentra
          cerca de 0 (normales). Las barras naranjas y rojas a la derecha representan las
          transacciones con mayor probabilidad de ser anómalas.
        </p>
        <canvas id="scoreDistChart"></canvas>
      </div>
    </div>
    <div class="grid-3" style="margin-top:1.5rem" id="methods-grid"></div>
  </section>

  <!-- DISTRIBUCIÓN -->
  <section id="distribucion" class="fade-in">
    <div class="section-title">&#128200; Distribuciones</div>
    <p class="section-desc">
      Visualización de cómo se distribuyen los montos, los niveles de riesgo asignados,
      los tipos de anomalía presentes en el dataset y la concordancia entre los tres
      métodos de detección.
    </p>
    <div class="grid-2">
      <div class="chart-card">
        <h3>Distribución de Montos (P1-P99)</h3>
        <p class="chart-desc">
          Histograma de montos excluyendo el 1% extremo superior e inferior para mejor
          visualización. Muestra la forma de la distribución: la mayoría de transacciones
          son montos bajos-medios, con cola hacia la derecha.
        </p>
        <canvas id="histogramChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Nivel de Riesgo</h3>
        <p class="chart-desc">
          Clasificación final de todas las transacciones. <strong>Crítico</strong> (score &ge; 0.75):
          requiere bloqueo inmediato. <strong>Alto</strong> (0.50-0.75): escalar a supervisión.
          <strong>Medio</strong> (0.25-0.50): requiere revisión. <strong>Bajo</strong> (&lt; 0.25): normal.
        </p>
        <canvas id="riskChart"></canvas>
      </div>
    </div>
    <div class="grid-2" style="margin-top:1.5rem">
      <div class="chart-card">
        <h3>Tipos de Anomalía (Ground Truth)</h3>
        <p class="chart-desc">
          Desglose de las anomalías reales inyectadas en el dataset sintético: montos atípicos,
          montos negativos, horarios inusuales, ráfagas de frecuencia, duplicados y cuentas nuevas.
          Sirve para evaluar qué tipos de fraude detecta mejor el sistema.
        </p>
        <canvas id="anomalyTypesChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Concordancia de Métodos</h3>
        <p class="chart-desc">
          Muestra cuántas transacciones fueron detectadas por cada combinación de métodos.
          Las barras de la derecha (2+ métodos) son las que se marcan como anomalías finales.
          La barra "Los 3" indica las detecciones con mayor confianza.
        </p>
        <canvas id="methodsBarChart"></canvas>
      </div>
    </div>
  </section>

  <!-- TIMELINE -->
  <section id="timeline" class="fade-in">
    <div class="section-title">&#128197; Evolución Temporal</div>
    <p class="section-desc">
      Volumen de transacciones y anomalías detectadas a lo largo del año.
      Las barras azules muestran el total mensual de operaciones y la línea roja
      superpuesta muestra cuántas de esas transacciones fueron marcadas como anómalas.
      Picos en la línea roja podrían indicar periodos de mayor actividad fraudulenta.
    </p>
    <div class="chart-card">
      <h3>Transacciones y Anomalías por Mes</h3>
      <canvas id="timelineChart"></canvas>
    </div>
  </section>

  <!-- HEATMAP -->
  <section id="heatmap" class="fade-in">
    <div class="section-title">&#128293; Heatmap de Anomalías (Hora &times; Día)</div>
    <p class="section-desc">
      Mapa de calor que cruza la hora del día (eje vertical, 00:00 a 23:00) con el día
      de la semana (eje horizontal). Las celdas más rojas concentran mayor cantidad de
      anomalías. Permite identificar patrones temporales: por ejemplo, operaciones
      sospechosas en madrugada o fines de semana, cuando normalmente no hay actividad.
    </p>
    <div class="chart-card">
      <div id="heatmap-container"></div>
    </div>
  </section>

  <!-- ALERTAS -->
  <section id="alertas" class="fade-in">
    <div class="section-title">&#128680; Tabla de Alertas</div>
    <p class="section-desc">
      Listado completo de todas las transacciones marcadas como anómalas, ordenadas por
      score de mayor a menor. Puedes filtrar por nivel de riesgo, buscar por ID, destinatario
      o RUT, y ordenar por cualquier columna. La columna <strong>Factores</strong> detalla
      qué señales dispararon la alerta (ej: "Z-Score elevado", "Horario nocturno", etc.).
    </p>
    <div class="controls">
      <input type="text" class="search-input" id="searchInput" placeholder="Buscar por ID, destinatario, RUT...">
      <button class="filter-btn active" data-filter="ALL">Todos</button>
      <button class="filter-btn" data-filter="CRITICO">Crítico</button>
      <button class="filter-btn" data-filter="ALTO">Alto</button>
      <button class="filter-btn" data-filter="MEDIO">Medio</button>
    </div>
    <div class="chart-card">
      <div class="table-wrapper">
        <table class="data-table" id="alertsTable">
          <thead>
            <tr>
              <th data-sort="id">ID</th>
              <th data-sort="fecha">Fecha</th>
              <th data-sort="hora">Hora</th>
              <th data-sort="destinatario">Destinatario</th>
              <th data-sort="categoria">Categoría</th>
              <th data-sort="monto">Monto</th>
              <th data-sort="score">Score</th>
              <th data-sort="nivel">Riesgo</th>
              <th>Factores</th>
            </tr>
          </thead>
          <tbody id="alertsBody"></tbody>
        </table>
      </div>
      <div class="pagination" id="pagination"></div>
    </div>
  </section>

</div>

  <!-- ACERCA DE -->
  <section id="acerca" class="fade-in">
    <div class="section-title">&#9881; Acerca de este Proyecto</div>
    <div class="grid-2">
      <div class="chart-card" style="line-height:1.8">
        <h3 style="color:var(--accent-blue);margin-bottom:0.8rem">Contexto y Objetivo</h3>
        <p style="color:var(--text-secondary);font-size:0.88rem">
          Este es un proyecto de demostración de capacidades de <strong style="color:var(--text-primary)">análisis
          de datos avanzado y detección de anomalías</strong> aplicado al dominio financiero.
        </p>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-top:0.8rem">
          El pipeline genera un dataset sintético de ~5.000 transacciones que simula movimientos
          de cuentas corrientes de una empresa chilena. Se inyectan intencionalmente 6 tipos de
          anomalías (montos atípicos, horarios fuera de patrón, ráfagas, duplicados, montos
          negativos y cuentas nuevas con montos altos) con etiquetas para poder medir la
          efectividad de la detección.
        </p>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-top:0.8rem">
          <strong style="color:var(--text-primary)">Todos los datos son 100% sintéticos.</strong>
          No se utiliza información real de ninguna empresa o persona.
        </p>
      </div>
      <div class="chart-card" style="line-height:1.8">
        <h3 style="color:var(--accent-purple);margin-bottom:0.8rem">Métodos de Detección</h3>
        <p style="color:var(--text-secondary);font-size:0.88rem">
          <strong style="color:var(--accent-blue)">Z-Score (peso 25%)</strong> &mdash;
          Mide cuántas desviaciones estándar se aleja el monto de una transacción respecto
          a la media global. Simple y efectivo para outliers extremos.
        </p>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-top:0.6rem">
          <strong style="color:var(--accent-green)">IQR (peso 25%)</strong> &mdash;
          Método de Tukey basado en el rango intercuartílico. Más robusto que Z-Score cuando
          la distribución de montos es asimétrica (como suele ser en datos financieros).
        </p>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-top:0.6rem">
          <strong style="color:var(--accent-purple)">Isolation Forest (peso 50%)</strong> &mdash;
          Algoritmo de machine learning no supervisado (scikit-learn) que analiza 7 variables
          simultáneamente: monto, hora, día, frecuencia del destinatario, desviación respecto
          a la categoría, si es horario nocturno y si es fin de semana.
        </p>
        <p style="color:var(--text-secondary);font-size:0.88rem;margin-top:0.8rem">
          Una transacción se marca como anómala cuando <strong style="color:var(--accent-red)">
          al menos 2 de los 3 métodos</strong> la detectan, reduciendo falsos positivos.
        </p>
      </div>
    </div>
    <div class="chart-card" style="margin-top:1.5rem;line-height:1.8">
      <h3 style="color:var(--accent-cyan);margin-bottom:0.8rem">Stack Tecnológico</h3>
      <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.5rem">
        <span class="pipeline-step"><strong>Python 3.12</strong></span>
        <span class="pipeline-step"><strong>pandas</strong> &mdash; manipulación de datos</span>
        <span class="pipeline-step"><strong>numpy</strong> &mdash; cálculos estadísticos</span>
        <span class="pipeline-step"><strong>scikit-learn</strong> &mdash; Isolation Forest</span>
        <span class="pipeline-step"><strong>Chart.js 4</strong> &mdash; visualización</span>
        <span class="pipeline-step"><strong>pytest</strong> &mdash; 33 tests</span>
        <span class="pipeline-step"><strong>GitHub Actions</strong> &mdash; CI/CD</span>
        <span class="pipeline-step"><strong>GitHub Pages</strong> &mdash; deploy</span>
      </div>
    </div>
  </section>

</div>

<footer>
  Detector de Anomalías Financieras &mdash;
  <a href="https://github.com/mechjook" target="_blank">@mechjook</a>
  &mdash; Datos 100% sintéticos
</footer>

<script>
const DATA = {chart_data_json};
const ITEMS_PER_PAGE = 20;
let currentPage = 1;
let currentFilter = "ALL";
let searchTerm = "";
let sortCol = "score";
let sortAsc = false;

// ===== KPIs =====
function renderKPIs() {{
  const k = DATA.kpis;
  const kpis = [
    {{ label: "Total Transacciones", value: k.total, fmt: "int", color: "var(--accent-blue)" }},
    {{ label: "Anomalías Detectadas", value: k.detected, fmt: "int", color: "var(--accent-red)",
      detail: `${{(k.detected/k.total*100).toFixed(1)}}% del total` }},
    {{ label: "Riesgo Crítico", value: k.critico, fmt: "int", color: "var(--accent-red)",
      detail: `+ ${{k.alto}} alto` }},
    {{ label: "Riesgo Alto", value: k.alto, fmt: "int", color: "var(--accent-orange)" }},
    {{ label: "Precision", value: k.precision, fmt: "pct", color: "var(--accent-green)",
      detail: `TP: ${{k.true_positives}} | FP: ${{k.false_positives}}` }},
    {{ label: "Recall", value: k.recall, fmt: "pct", color: "var(--accent-cyan)",
      detail: `TP: ${{k.true_positives}} | FN: ${{k.false_negatives}}` }},
    {{ label: "F1-Score", value: k.f1, fmt: "pct", color: "var(--accent-purple)" }},
    {{ label: "Monto en Anomalías", value: k.monto_anomalo, fmt: "money", color: "var(--accent-yellow)",
      detail: `${{(Math.abs(k.monto_anomalo)/Math.abs(k.monto_total)*100).toFixed(1)}}% del total` }},
  ];

  const grid = document.getElementById("kpi-grid");
  kpis.forEach((kpi, i) => {{
    const card = document.createElement("div");
    card.className = "kpi-card";
    card.style.animationDelay = `${{i * 0.05}}s`;
    const fmtValue = kpi.fmt === "int" ? formatInt(kpi.value)
      : kpi.fmt === "pct" ? (kpi.value * 100).toFixed(1) + "%"
      : formatMoney(kpi.value);
    card.innerHTML = `
      <div class="kpi-label">${{kpi.label}}</div>
      <div class="kpi-value" style="color:${{kpi.color}}">${{fmtValue}}</div>
      ${{kpi.detail ? `<div class="kpi-detail">${{kpi.detail}}</div>` : ""}}
    `;
    grid.appendChild(card);
  }});
}}

// ===== CHARTS =====
function renderCharts() {{
  const chartDefaults = {{
    color: "#94A3B8",
    borderColor: "#334155",
  }};
  Chart.defaults.color = chartDefaults.color;
  Chart.defaults.borderColor = chartDefaults.borderColor;

  // Scatter
  new Chart(document.getElementById("scatterChart"), {{
    type: "scatter",
    data: {{
      datasets: [
        {{
          label: "Normal",
          data: DATA.scatter.normal.x.map((x, i) => ({{ x, y: DATA.scatter.normal.y[i] }})),
          backgroundColor: "rgba(59,130,246,0.3)",
          borderColor: "rgba(59,130,246,0.6)",
          pointRadius: 2,
        }},
        {{
          label: "Anomalía",
          data: DATA.scatter.anomaly.x.map((x, i) => ({{ x, y: DATA.scatter.anomaly.y[i] }})),
          backgroundColor: "rgba(239,68,68,0.5)",
          borderColor: "rgba(239,68,68,0.8)",
          pointRadius: 3,
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        x: {{ title: {{ display: true, text: "Score" }} }},
        y: {{ title: {{ display: true, text: "Monto ($)" }},
              ticks: {{ callback: v => "$" + formatInt(v) }} }}
      }}
    }}
  }});

  // Score Distribution
  new Chart(document.getElementById("scoreDistChart"), {{
    type: "bar",
    data: {{
      labels: DATA.score_distribution.labels,
      datasets: [{{
        label: "Transacciones",
        data: DATA.score_distribution.counts,
        backgroundColor: DATA.score_distribution.labels.map(l =>
          parseFloat(l) >= 0.5 ? "rgba(239,68,68,0.6)" :
          parseFloat(l) >= 0.25 ? "rgba(249,115,22,0.6)" :
          "rgba(59,130,246,0.4)"
        ),
        borderWidth: 0,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ title: {{ display: true, text: "Score" }},
              ticks: {{ maxTicksLimit: 10 }} }},
        y: {{ title: {{ display: true, text: "Cantidad" }} }}
      }}
    }}
  }});

  // Histogram
  new Chart(document.getElementById("histogramChart"), {{
    type: "bar",
    data: {{
      labels: DATA.histogram.labels,
      datasets: [{{
        label: "Frecuencia",
        data: DATA.histogram.counts,
        backgroundColor: "rgba(6,182,212,0.5)",
        borderColor: "rgba(6,182,212,0.8)",
        borderWidth: 1,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 8 }} }},
        y: {{ title: {{ display: true, text: "Frecuencia" }} }}
      }}
    }}
  }});

  // Risk Doughnut
  new Chart(document.getElementById("riskChart"), {{
    type: "doughnut",
    data: {{
      labels: DATA.risk_distribution.labels,
      datasets: [{{
        data: DATA.risk_distribution.counts,
        backgroundColor: DATA.risk_distribution.colors,
        borderWidth: 2,
        borderColor: "#1E293B",
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ position: "bottom" }},
      }},
      cutout: "60%",
    }}
  }});

  // Anomaly Types
  if (DATA.anomaly_types.labels.length > 0) {{
    new Chart(document.getElementById("anomalyTypesChart"), {{
      type: "bar",
      data: {{
        labels: DATA.anomaly_types.labels.map(l => l.replace(/_/g, " ")),
        datasets: [{{
          label: "Cantidad",
          data: DATA.anomaly_types.counts,
          backgroundColor: [
            "rgba(239,68,68,0.6)", "rgba(249,115,22,0.6)",
            "rgba(234,179,8,0.6)", "rgba(168,85,247,0.6)",
            "rgba(6,182,212,0.6)", "rgba(59,130,246,0.6)",
          ],
          borderWidth: 0,
        }}]
      }},
      options: {{
        indexAxis: "y",
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
      }}
    }});
  }}

  // Methods comparison bar
  const dm = DATA.detection_methods;
  new Chart(document.getElementById("methodsBarChart"), {{
    type: "bar",
    data: {{
      labels: ["Solo Z-Score", "Solo IQR", "Solo IForest", "Z+IQR", "Z+IF", "IQR+IF", "Los 3"],
      datasets: [{{
        label: "Transacciones",
        data: [dm.zscore_only, dm.iqr_only, dm.iforest_only,
               dm.zscore_iqr, dm.zscore_iforest, dm.iqr_iforest, dm.all_three],
        backgroundColor: [
          "rgba(59,130,246,0.5)", "rgba(34,197,94,0.5)", "rgba(168,85,247,0.5)",
          "rgba(6,182,212,0.5)", "rgba(249,115,22,0.5)", "rgba(234,179,8,0.5)",
          "rgba(239,68,68,0.7)"
        ],
        borderWidth: 0,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: "Cantidad" }} }}
      }}
    }}
  }});

  // Timeline
  new Chart(document.getElementById("timelineChart"), {{
    type: "bar",
    data: {{
      labels: DATA.timeline.labels,
      datasets: [
        {{
          label: "Total transacciones",
          data: DATA.timeline.total,
          backgroundColor: "rgba(59,130,246,0.4)",
          borderColor: "rgba(59,130,246,0.8)",
          borderWidth: 1,
          order: 2,
        }},
        {{
          label: "Anomalías detectadas",
          type: "line",
          data: DATA.timeline.anomalias,
          borderColor: "rgba(239,68,68,0.9)",
          backgroundColor: "rgba(239,68,68,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 4,
          order: 1,
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        y: {{ title: {{ display: true, text: "Cantidad" }} }}
      }}
    }}
  }});
}}

// ===== METHODS GRID =====
function renderMethodsGrid() {{
  const dm = DATA.detection_methods;
  const methods = [
    {{ name: "Z-Score", count: dm.zscore_total, color: "var(--accent-blue)", icon: "&#963;" }},
    {{ name: "IQR", count: dm.iqr_total, color: "var(--accent-green)", icon: "&#8800;" }},
    {{ name: "Isolation Forest", count: dm.iforest_total, color: "var(--accent-purple)", icon: "&#127794;" }},
  ];
  const grid = document.getElementById("methods-grid");
  methods.forEach(m => {{
    const card = document.createElement("div");
    card.className = "method-card";
    const pct = (m.count / DATA.kpis.total * 100).toFixed(1);
    card.innerHTML = `
      <div class="method-name" style="color:${{m.color}}">${{m.icon}} ${{m.name}}</div>
      <div class="method-count" style="color:${{m.color}}">${{formatInt(m.count)}}</div>
      <div class="kpi-detail">${{pct}}% flaggeadas</div>
      <div class="progress-bg"><div class="progress-fill" style="width:${{pct}}%;background:${{m.color}}"></div></div>
    `;
    grid.appendChild(card);
  }});
}}

// ===== HEATMAP =====
function renderHeatmap() {{
  const hm = DATA.heatmap;
  const container = document.getElementById("heatmap-container");
  let maxVal = 0;
  hm.data.forEach(row => row.forEach(v => {{ if (v > maxVal) maxVal = v; }}));

  let html = '<div class="heatmap-grid">';
  html += '<div class="heatmap-header"></div>';
  hm.days.forEach(d => {{ html += `<div class="heatmap-header">${{d}}</div>`; }});

  for (let h = 0; h < 24; h++) {{
    html += `<div class="heatmap-header">${{hm.hours[h]}}</div>`;
    for (let d = 0; d < 7; d++) {{
      const val = hm.data[h][d];
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const r = Math.round(239 * intensity);
      const g = Math.round(68 * intensity);
      const b = Math.round(68 * intensity + 30 * (1 - intensity));
      const bg = val > 0 ? `rgba(${{r}},${{g}},${{b}},0.8)` : "rgba(30,41,59,0.5)";
      html += `<div class="heatmap-cell" style="background:${{bg}}" title="${{hm.hours[h]}} ${{hm.days[d]}}: ${{val}} anomalías">${{val || ""}}</div>`;
    }}
  }}
  html += '</div>';
  container.innerHTML = html;
}}

// ===== TABLE =====
function getFilteredData() {{
  let items = DATA.all_anomalies;
  if (currentFilter !== "ALL") {{
    items = items.filter(a => a.nivel === currentFilter);
  }}
  if (searchTerm) {{
    const st = searchTerm.toLowerCase();
    items = items.filter(a =>
      a.id.toLowerCase().includes(st) ||
      a.destinatario.toLowerCase().includes(st) ||
      a.rut.toLowerCase().includes(st) ||
      a.categoria.toLowerCase().includes(st) ||
      a.sucursal.toLowerCase().includes(st)
    );
  }}
  items.sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === "string") {{ va = va.toLowerCase(); vb = vb.toLowerCase(); }}
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  }});
  return items;
}}

function renderTable() {{
  const items = getFilteredData();
  const totalPages = Math.ceil(items.length / ITEMS_PER_PAGE);
  if (currentPage > totalPages) currentPage = totalPages || 1;
  const start = (currentPage - 1) * ITEMS_PER_PAGE;
  const pageItems = items.slice(start, start + ITEMS_PER_PAGE);

  const tbody = document.getElementById("alertsBody");
  tbody.innerHTML = pageItems.map(a => `
    <tr>
      <td style="font-family:monospace">${{a.id}}</td>
      <td>${{a.fecha}}</td>
      <td>${{a.hora}}</td>
      <td>${{a.destinatario}}</td>
      <td>${{a.categoria}}</td>
      <td style="text-align:right;font-variant-numeric:tabular-nums">${{formatMoney(a.monto)}}</td>
      <td style="text-align:center;font-weight:700">${{a.score.toFixed(3)}}</td>
      <td><span class="risk-badge" style="background:${{a.color}}20;color:${{a.color}}">${{a.nivel}}</span></td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${{a.factores}}">${{a.factores}}</td>
    </tr>
  `).join("");

  const pag = document.getElementById("pagination");
  if (totalPages <= 1) {{ pag.innerHTML = ""; return; }}
  let pagHtml = `<button class="page-btn" onclick="goPage(1)" ${{currentPage===1?"disabled":""}}>&laquo;</button>`;
  pagHtml += `<button class="page-btn" onclick="goPage(${{currentPage-1}})" ${{currentPage===1?"disabled":""}}>&lsaquo;</button>`;
  const maxVisible = 5;
  let startPage = Math.max(1, currentPage - Math.floor(maxVisible/2));
  let endPage = Math.min(totalPages, startPage + maxVisible - 1);
  if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);
  for (let p = startPage; p <= endPage; p++) {{
    pagHtml += `<button class="page-btn ${{p===currentPage?'active':''}}" onclick="goPage(${{p}})">${{p}}</button>`;
  }}
  pagHtml += `<span class="page-info">${{items.length}} registros</span>`;
  pagHtml += `<button class="page-btn" onclick="goPage(${{currentPage+1}})" ${{currentPage===totalPages?"disabled":""}}>&rsaquo;</button>`;
  pagHtml += `<button class="page-btn" onclick="goPage(${{totalPages}})" ${{currentPage===totalPages?"disabled":""}}>&raquo;</button>`;
  pag.innerHTML = pagHtml;
}}

function goPage(p) {{ currentPage = p; renderTable(); }}

// ===== EVENT LISTENERS =====
document.getElementById("searchInput").addEventListener("input", e => {{
  searchTerm = e.target.value; currentPage = 1; renderTable();
}});

document.querySelectorAll(".filter-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    currentPage = 1;
    renderTable();
  }});
}});

document.querySelectorAll("[data-sort]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col = th.dataset.sort;
    if (sortCol === col) {{ sortAsc = !sortAsc; }}
    else {{ sortCol = col; sortAsc = true; }}
    renderTable();
  }});
}});

// Navbar active section
const sections = document.querySelectorAll("section");
const navLinks = document.querySelectorAll(".nav-links a");
window.addEventListener("scroll", () => {{
  let current = "";
  sections.forEach(s => {{
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  }});
  navLinks.forEach(a => {{
    a.classList.toggle("active", a.getAttribute("href") === "#" + current);
  }});
}});

// ===== FORMATTERS =====
function formatInt(n) {{ return Math.round(n).toLocaleString("es-CL"); }}
function formatMoney(n) {{ return "$" + Math.round(n).toLocaleString("es-CL"); }}

// ===== INIT =====
renderKPIs();
renderCharts();
renderMethodsGrid();
renderHeatmap();
renderTable();
</script>
</body>
</html>'''
