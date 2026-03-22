"""
Validador de Datos — Verifica integridad del dataset de transacciones.

Realiza validaciones de:
  - Estructura del CSV (columnas requeridas, tipos)
  - Integridad de valores (RUTs, fechas, montos)
  - Consistencia lógica (sin IDs duplicados, fechas en rango)

Autor: José Nicolás Candia (@mechjook)
"""

import os
import re
from dataclasses import dataclass, field

import pandas as pd


REQUIRED_COLUMNS = [
    "id_transaccion", "fecha", "hora", "dia_semana", "cuenta_origen",
    "cuenta_nombre", "destinatario_rut", "destinatario_nombre", "categoria",
    "monto", "sucursal", "canal", "descripcion", "es_anomalia", "tipo_anomalia",
]

RUT_PATTERN = re.compile(r"^\d{1,2}\.\d{3}\.\d{3}-[\dkK]$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
TXN_ID_PATTERN = re.compile(r"^TXN-\d{6}$")


@dataclass
class ValidationResult:
    """Resultado de la validación con errores y advertencias."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_file(path: str) -> ValidationResult:
    """Valida que el archivo existe y es legible como CSV."""
    result = ValidationResult()

    if not os.path.exists(path):
        result.add_error(f"Archivo no encontrado: {path}")
        return result

    try:
        df = pd.read_csv(path, encoding="utf-8", nrows=5)
    except Exception as e:
        result.add_error(f"Error leyendo CSV: {e}")
        return result

    # Validar columnas requeridas
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        result.add_error(f"Columnas faltantes: {', '.join(sorted(missing))}")

    return result


def validate_data(df: pd.DataFrame) -> ValidationResult:
    """Valida integridad y consistencia del dataset completo."""
    result = ValidationResult()
    result.stats["total_rows"] = len(df)

    # --- Columnas ---
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        result.add_error(f"Columnas faltantes: {', '.join(sorted(missing))}")
        return result

    # --- IDs únicos ---
    dup_ids = df["id_transaccion"].duplicated().sum()
    if dup_ids > 0:
        result.add_error(f"IDs duplicados: {dup_ids}")
    result.stats["unique_ids"] = df["id_transaccion"].nunique()

    # --- Formato de ID ---
    invalid_ids = df[~df["id_transaccion"].str.match(TXN_ID_PATTERN, na=False)]
    if len(invalid_ids) > 0:
        result.add_warning(f"IDs con formato inválido: {len(invalid_ids)}")

    # --- Fechas ---
    invalid_dates = df[~df["fecha"].str.match(DATE_PATTERN, na=False)]
    if len(invalid_dates) > 0:
        result.add_error(f"Fechas con formato inválido: {len(invalid_dates)}")
    result.stats["date_range"] = f"{df['fecha'].min()} a {df['fecha'].max()}"

    # --- Horas ---
    invalid_times = df[~df["hora"].str.match(TIME_PATTERN, na=False)]
    if len(invalid_times) > 0:
        result.add_error(f"Horas con formato inválido: {len(invalid_times)}")

    # --- RUTs ---
    invalid_ruts = df[~df["destinatario_rut"].str.match(RUT_PATTERN, na=False)]
    if len(invalid_ruts) > 0:
        result.add_warning(f"RUTs con formato no estándar: {len(invalid_ruts)}")
    result.stats["unique_ruts"] = df["destinatario_rut"].nunique()

    # --- Montos ---
    try:
        montos = pd.to_numeric(df["monto"], errors="coerce")
        nulls = montos.isna().sum()
        if nulls > 0:
            result.add_error(f"Montos no numéricos: {nulls}")
        result.stats["monto_min"] = float(montos.min())
        result.stats["monto_max"] = float(montos.max())
        result.stats["monto_mean"] = float(montos.mean())
    except Exception as e:
        result.add_error(f"Error procesando montos: {e}")

    # --- Nulls ---
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        result.add_warning(f"Valores nulos totales: {total_nulls}")
        for col, count in null_counts[null_counts > 0].items():
            result.add_warning(f"  - {col}: {count} nulos")

    # --- Etiquetas de anomalía ---
    anomaly_count = df["es_anomalia"].sum()
    result.stats["anomalias_etiquetadas"] = int(anomaly_count)
    result.stats["tasa_anomalia"] = anomaly_count / len(df) if len(df) > 0 else 0

    tipos = df["tipo_anomalia"].value_counts().to_dict()
    result.stats["tipos_anomalia"] = tipos

    return result


def validate_all(path: str) -> ValidationResult:
    """Ejecuta validación completa de archivo + datos."""
    print("\n" + "=" * 60)
    print("ETAPA 1: VALIDACIÓN DE DATOS")
    print("=" * 60)

    # Validar archivo
    file_result = validate_file(path)
    if not file_result.is_valid:
        for err in file_result.errors:
            print(f"  ✗ {err}")
        return file_result

    # Leer y validar datos
    df = pd.read_csv(path, encoding="utf-8")
    result = validate_data(df)

    # Imprimir resultados
    if result.errors:
        print("  Errores:")
        for err in result.errors:
            print(f"    ✗ {err}")
    if result.warnings:
        print("  Advertencias:")
        for warn in result.warnings:
            print(f"    ⚠ {warn}")

    print(f"\n  Estadísticas:")
    print(f"    Registros totales  : {result.stats.get('total_rows', 0):,}")
    print(f"    IDs únicos         : {result.stats.get('unique_ids', 0):,}")
    print(f"    RUTs únicos        : {result.stats.get('unique_ruts', 0):,}")
    print(f"    Rango de fechas    : {result.stats.get('date_range', 'N/A')}")
    print(f"    Monto mín/máx      : ${result.stats.get('monto_min', 0):,.0f} / "
          f"${result.stats.get('monto_max', 0):,.0f}")
    print(f"    Anomalías etiquetadas: {result.stats.get('anomalias_etiquetadas', 0):,}")

    status = "✓ VÁLIDO" if result.is_valid else "✗ INVÁLIDO"
    print(f"\n  Estado: {status}")
    return result
