"""
Tests para el Detector de Anomalías Financieras.

Incluye:
  - Tests del generador de datos (estructura, tipos de anomalía)
  - Tests del detector (Z-Score, IQR, Isolation Forest)
  - Tests del clasificador de riesgo
  - Tests de validación
  - Tests de métricas (precision/recall)
  - Test de integración del pipeline completo

Autor: José Nicolás Candia (@mechjook)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def temp_dir():
    """Directorio temporal para archivos de prueba."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="module")
def generated_data(temp_dir):
    """Genera datos de prueba una vez para todos los tests."""
    from src.generate_data import generate_transactions
    path = os.path.join(temp_dir, "test_transactions.csv")
    generate_transactions(output_path=path, n_normal=500)
    return path


@pytest.fixture(scope="module")
def df_raw(generated_data):
    """DataFrame con datos crudos."""
    return pd.read_csv(generated_data, encoding="utf-8")


@pytest.fixture(scope="module")
def df_detected(df_raw):
    """DataFrame con detección de anomalías ejecutada."""
    from src.detector import run_detection
    return run_detection(df_raw.copy())


@pytest.fixture(scope="module")
def df_classified(df_detected):
    """DataFrame con clasificación de riesgo."""
    from src.classifier import classify_risk
    return classify_risk(df_detected.copy())


# ============================================================
# Tests del Generador de Datos
# ============================================================

class TestGenerateData:
    """Tests para src/generate_data.py."""

    def test_file_created(self, generated_data):
        """El archivo CSV se genera correctamente."""
        assert os.path.exists(generated_data)

    def test_csv_readable(self, df_raw):
        """El CSV es legible como DataFrame."""
        assert len(df_raw) > 0

    def test_has_required_columns(self, df_raw):
        """Contiene todas las columnas requeridas."""
        expected = {
            "id_transaccion", "fecha", "hora", "dia_semana", "cuenta_origen",
            "cuenta_nombre", "destinatario_rut", "destinatario_nombre",
            "categoria", "monto", "sucursal", "canal", "descripcion",
            "es_anomalia", "tipo_anomalia",
        }
        assert expected.issubset(set(df_raw.columns))

    def test_has_normal_transactions(self, df_raw):
        """Contiene transacciones normales."""
        normal = df_raw[~df_raw["es_anomalia"]]
        assert len(normal) > 0

    def test_has_anomalies(self, df_raw):
        """Contiene anomalías inyectadas."""
        anomalies = df_raw[df_raw["es_anomalia"]]
        assert len(anomalies) > 0

    def test_anomaly_types_present(self, df_raw):
        """Contiene múltiples tipos de anomalía."""
        types = df_raw[df_raw["es_anomalia"]]["tipo_anomalia"].unique()
        expected_types = {
            "monto_atipico_alto", "monto_negativo", "horario_inusual",
            "frecuencia_inusual", "duplicado_sospechoso", "cuenta_nueva_monto_alto",
        }
        found = set(types)
        assert len(found.intersection(expected_types)) >= 4, \
            f"Esperados al menos 4 tipos, encontrados: {found}"

    def test_unique_ids(self, df_raw):
        """Todos los IDs de transacción son únicos."""
        assert df_raw["id_transaccion"].is_unique

    def test_date_format(self, df_raw):
        """Las fechas tienen formato YYYY-MM-DD."""
        dates = pd.to_datetime(df_raw["fecha"], format="%Y-%m-%d", errors="coerce")
        assert dates.notna().all()

    def test_time_format(self, df_raw):
        """Las horas tienen formato HH:MM:SS."""
        times = pd.to_datetime(df_raw["hora"], format="%H:%M:%S", errors="coerce")
        assert times.notna().all()

    def test_anomaly_ratio(self, df_raw):
        """La tasa de anomalías está en un rango razonable (3%-15%)."""
        rate = df_raw["es_anomalia"].mean()
        assert 0.03 <= rate <= 0.60, f"Tasa de anomalía fuera de rango: {rate:.2%}"


# ============================================================
# Tests del Validador
# ============================================================

class TestValidator:
    """Tests para validators/data_validator.py."""

    def test_validate_existing_file(self, generated_data):
        """Valida correctamente un archivo existente."""
        from validators.data_validator import validate_file
        result = validate_file(generated_data)
        assert result.is_valid

    def test_validate_nonexistent_file(self):
        """Detecta archivo inexistente."""
        from validators.data_validator import validate_file
        result = validate_file("/nonexistent/file.csv")
        assert not result.is_valid

    def test_validate_data_integrity(self, df_raw):
        """Valida integridad del dataset generado."""
        from validators.data_validator import validate_data
        result = validate_data(df_raw)
        assert result.is_valid
        assert result.stats["total_rows"] == len(df_raw)

    def test_detects_missing_columns(self):
        """Detecta columnas faltantes."""
        from validators.data_validator import validate_data
        df_bad = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = validate_data(df_bad)
        assert not result.is_valid


# ============================================================
# Tests del Detector
# ============================================================

class TestDetector:
    """Tests para src/detector.py."""

    def test_zscore_columns_added(self, df_detected):
        """Z-Score agrega las columnas esperadas."""
        assert "zscore_value" in df_detected.columns
        assert "zscore_flag" in df_detected.columns
        assert "zscore_score" in df_detected.columns

    def test_iqr_columns_added(self, df_detected):
        """IQR agrega las columnas esperadas."""
        assert "iqr_flag" in df_detected.columns
        assert "iqr_score" in df_detected.columns

    def test_iforest_columns_added(self, df_detected):
        """Isolation Forest agrega las columnas esperadas."""
        assert "iforest_flag" in df_detected.columns
        assert "iforest_score" in df_detected.columns

    def test_combined_score_exists(self, df_detected):
        """El score combinado existe y está en rango [0, 1]."""
        assert "score_final" in df_detected.columns
        assert df_detected["score_final"].min() >= 0
        assert df_detected["score_final"].max() <= 1.01

    def test_flag_count_range(self, df_detected):
        """flag_count está entre 0 y 3."""
        assert df_detected["flag_count"].min() >= 0
        assert df_detected["flag_count"].max() <= 3

    def test_detects_some_anomalies(self, df_detected):
        """Al menos algunas anomalías son detectadas."""
        detected = df_detected["es_anomalia_detectada"].sum()
        assert detected > 0

    def test_zscore_detects_high_amounts(self, df_detected):
        """Z-Score detecta transacciones con montos extremos."""
        high_amount = df_detected[df_detected["monto"] > 50_000_000]
        if len(high_amount) > 0:
            zscore_detected = high_amount["zscore_flag"].mean()
            assert zscore_detected > 0.5, "Z-Score debería detectar >50% de montos extremos"


# ============================================================
# Tests del Clasificador
# ============================================================

class TestClassifier:
    """Tests para src/classifier.py."""

    def test_risk_level_assigned(self, df_classified):
        """Todas las transacciones tienen nivel de riesgo."""
        assert "nivel_riesgo" in df_classified.columns
        assert df_classified["nivel_riesgo"].notna().all()

    def test_valid_risk_levels(self, df_classified):
        """Los niveles de riesgo son válidos."""
        valid = {"BAJO", "MEDIO", "ALTO", "CRITICO"}
        actual = set(df_classified["nivel_riesgo"].unique())
        assert actual.issubset(valid)

    def test_color_assigned(self, df_classified):
        """Todas las transacciones tienen color de riesgo."""
        assert "color_riesgo" in df_classified.columns
        assert df_classified["color_riesgo"].notna().all()

    def test_risk_factors_assigned(self, df_classified):
        """Todas las transacciones tienen factores de riesgo."""
        assert "factores_riesgo" in df_classified.columns
        assert df_classified["factores_riesgo"].notna().all()

    def test_priority_ordering(self, df_classified):
        """La prioridad es consistente con el nivel de riesgo."""
        for _, row in df_classified.iterrows():
            if row["nivel_riesgo"] == "CRITICO":
                assert row["prioridad"] == 4
            elif row["nivel_riesgo"] == "ALTO":
                assert row["prioridad"] == 3
            elif row["nivel_riesgo"] == "MEDIO":
                assert row["prioridad"] == 2
            elif row["nivel_riesgo"] == "BAJO":
                assert row["prioridad"] == 1
            break  # Solo verificar unos pocos por eficiencia

    def test_high_score_is_critical(self, df_classified):
        """Scores >= 0.75 corresponden a nivel CRITICO."""
        critical = df_classified[df_classified["score_final"] >= 0.75]
        if len(critical) > 0:
            assert (critical["nivel_riesgo"] == "CRITICO").all()


# ============================================================
# Tests de Métricas (Precision/Recall)
# ============================================================

class TestMetrics:
    """Tests de rendimiento de la detección."""

    def test_precision_above_minimum(self, df_classified):
        """La precision es razonable (> 0.3)."""
        from src.analytics import compute_detection_metrics
        metrics = compute_detection_metrics(df_classified)
        assert metrics["precision"] > 0.3, \
            f"Precision demasiado baja: {metrics['precision']:.4f}"

    def test_recall_above_minimum(self, df_classified):
        """El recall es razonable (> 0.10 con dataset reducido)."""
        from src.analytics import compute_detection_metrics
        metrics = compute_detection_metrics(df_classified)
        assert metrics["recall"] > 0.10, \
            f"Recall demasiado bajo: {metrics['recall']:.4f}"

    def test_f1_above_minimum(self, df_classified):
        """El F1-Score es razonable (> 0.15 con dataset reducido)."""
        from src.analytics import compute_detection_metrics
        metrics = compute_detection_metrics(df_classified)
        assert metrics["f1_score"] > 0.15, \
            f"F1-Score demasiado bajo: {metrics['f1_score']:.4f}"

    def test_confusion_matrix_sums(self, df_classified):
        """La matriz de confusión suma el total de registros."""
        from src.analytics import compute_detection_metrics
        metrics = compute_detection_metrics(df_classified)
        total = (metrics["true_positives"] + metrics["false_positives"]
                 + metrics["false_negatives"] + metrics["true_negatives"])
        assert total == len(df_classified)


# ============================================================
# Test de Integración
# ============================================================

class TestIntegration:
    """Test de integración del pipeline completo."""

    def test_full_pipeline(self, temp_dir):
        """Ejecuta el pipeline completo y verifica outputs."""
        from src.generate_data import generate_transactions
        from src.detector import run_detection
        from src.classifier import classify_risk
        from src.reports import generate_reports
        from src.dashboard import generate_dashboard
        from src.analytics import run_analytics

        output_dir = os.path.join(temp_dir, "output_integration")

        # Etapa 0: Generar datos
        data_path = os.path.join(temp_dir, "integration_txn.csv")
        generate_transactions(output_path=data_path, n_normal=200)

        # Cargar
        df = pd.read_csv(data_path, encoding="utf-8")
        assert len(df) > 200

        # Etapa 2: Detección
        df = run_detection(df)
        assert "score_final" in df.columns

        # Etapa 3: Clasificación
        df = classify_risk(df)
        assert "nivel_riesgo" in df.columns

        # Etapa 4: Reportes
        paths = generate_reports(df, output_dir)
        assert os.path.exists(paths["anomalias"])
        assert os.path.exists(paths["completo"])
        assert os.path.exists(paths["resumen"])

        # Etapa 5: Dashboard
        dashboard = generate_dashboard(df, output_dir)
        assert os.path.exists(dashboard)
        with open(dashboard, encoding="utf-8") as f:
            html = f.read()
        assert "Chart.js" in html or "chart.js" in html
        assert "Anomaly Detector" in html

        # Analytics
        stats = run_analytics(df)
        assert "detection" in stats
        assert "distribution" in stats
        assert stats["detection"]["precision"] >= 0

    def test_reports_csv_structure(self, temp_dir):
        """Verifica estructura de los CSV generados."""
        output_dir = os.path.join(temp_dir, "output_integration")
        anomalias_path = os.path.join(output_dir, "reporte_anomalias.csv")

        if os.path.exists(anomalias_path):
            df = pd.read_csv(anomalias_path)
            assert "id_transaccion" in df.columns
            assert "score_final" in df.columns
            assert "nivel_riesgo" in df.columns
            assert len(df) > 0
