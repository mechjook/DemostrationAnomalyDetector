# Detector de Anomalías Financieras

Sistema automatizado que analiza transacciones financieras sintéticas, detecta anomalías usando múltiples métodos estadísticos y de machine learning, clasifica el riesgo y publica un dashboard interactivo con los hallazgos.

[![Anomaly Detection Pipeline](https://github.com/mechjook/DemostrationAnomalyDetector/actions/workflows/anomaly_pipeline.yml/badge.svg)](https://github.com/mechjook/DemostrationAnomalyDetector/actions/workflows/anomaly_pipeline.yml)

## Dashboard

Disponible en: **[GitHub Pages](https://mechjook.github.io/DemostrationAnomalyDetector/)**

## Arquitectura del Pipeline

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────────────────────┐
│  Generación      │───▶│  Validación   │───▶│  Detección de Anomalías      │
│  ~5000 TXNs      │    │  Estructura   │    │  Z-Score + IQR + IForest     │
└─────────────────┘    └──────────────┘    └──────────────┬───────────────┘
                                                          │
┌─────────────────┐    ┌──────────────┐    ┌──────────────▼───────────────┐
│  Dashboard       │◀──│  Reportes     │◀──│  Clasificación de Riesgo     │
│  HTML + Chart.js │    │  CSV          │    │  Bajo/Medio/Alto/Crítico     │
└─────────────────┘    └──────────────┘    └──────────────────────────────┘
```

## Etapas

| # | Etapa | Descripción |
|---|-------|-------------|
| 0 | **Generación** | Crea ~5000 transacciones sintéticas con 6 tipos de anomalías inyectadas |
| 1 | **Validación** | Verifica estructura, formatos, integridad y consistencia del dataset |
| 2 | **Detección** | Aplica Z-Score, IQR y Isolation Forest; combina scores ponderados |
| 3 | **Clasificación** | Asigna nivel de riesgo (bajo/medio/alto/crítico) con factores legibles |
| 4 | **Reportes** | Genera CSV de anomalías, reporte completo y resumen ejecutivo |
| 5 | **Dashboard** | Página HTML interactiva con 7+ gráficos, heatmap y tabla filtrable |

## Tipos de Anomalías Inyectadas

| Tipo | Descripción | Cantidad |
|------|-------------|----------|
| `monto_atipico_alto` | Montos 5x-20x por encima del rango normal | ~80 |
| `monto_negativo` | Reversiones sospechosas con montos negativos | ~30 |
| `horario_inusual` | Transacciones en madrugada o fines de semana | ~60 |
| `frecuencia_inusual` | Ráfagas de transacciones en segundos | ~60 |
| `duplicado_sospechoso` | Mismo monto y destinatario, segundos después | ~40 |
| `cuenta_nueva_monto_alto` | Destinatarios nuevos con montos altos | ~40 |

## Métodos de Detección

| Método | Peso | Enfoque |
|--------|------|---------|
| **Z-Score** | 25% | Distancia estadística del monto respecto a la media (umbral: 3σ) |
| **IQR** | 25% | Rango intercuartílico de Tukey (k=1.5) para outliers robustos |
| **Isolation Forest** | 50% | ML no supervisado con 7 features (monto, hora, día, frecuencia, etc.) |

Criterio de detección: una transacción se marca como anómala si **al menos 2 de 3 métodos** la detectan.

## Ejecución Local

```bash
pip install -r requirements.txt
python main.py
```

## Tests

```bash
pytest tests/ -v
```

Los tests validan:
- Estructura del dataset generado (columnas, tipos, formatos)
- Funcionamiento de cada método de detección
- Clasificación de riesgo correcta
- Precision > 0.3, Recall > 0.3, F1 > 0.3
- Pipeline de integración completo

## CI/CD

El workflow de GitHub Actions ejecuta:
1. **Tests** — pytest con validación completa
2. **Pipeline** — genera datos, ejecuta detección, produce dashboard
3. **Deploy** — publica el dashboard en GitHub Pages (solo en `main`)

## Stack

- Python 3.12
- pandas / numpy
- scikit-learn (Isolation Forest)
- Chart.js 4
- pytest
- GitHub Actions + GitHub Pages

## Autor

**José Nicolás Candia** — [@mechjook](https://github.com/mechjook)
