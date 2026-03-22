"""
Generador de datos sintéticos — Transacciones financieras con anomalías.

Produce un dataset de ~5000 transacciones financieras realistas con
anomalías inyectadas intencionalmente para demostrar capacidades de
detección. Cada anomalía queda etiquetada para validar precision/recall.

Tipos de anomalías inyectadas:
  - Montos atípicos (extremadamente altos o negativos)
  - Frecuencia inusual (ráfagas de transacciones en minutos)
  - Horarios fuera de patrón (madrugada, fines de semana)
  - Duplicados sospechosos (mismo monto/destinatario en segundos)
  - Cuentas nuevas con montos altos (first-seen risk)

Autor: José Nicolás Candia (@mechjook)
"""

import csv
import os
import random
from datetime import datetime, timedelta
from typing import Any

SEED = 42
random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# --- Catálogos de datos realistas ---

CUENTAS_ORIGEN = [
    ("CC-1001", "Caja Principal"),
    ("CC-1002", "Banco Estado CTA CTE"),
    ("CC-1003", "Banco Chile CTA CTE"),
    ("CC-1004", "Banco Santander CTA CTE"),
    ("CC-1005", "Caja Chica Operaciones"),
    ("CC-1006", "Banco BCI CTA CTE"),
    ("CC-1007", "Fondo Rotativo"),
]

DESTINATARIOS = [
    ("76.123.456-7", "Proveedora Industrial SpA"),
    ("76.234.567-8", "Servicios Logísticos Ltda"),
    ("76.345.678-9", "Importadora del Pacífico SA"),
    ("76.456.789-0", "Comercial Austral SpA"),
    ("76.567.890-1", "Tecnología y Redes Ltda"),
    ("76.678.901-2", "Distribuidora Central SA"),
    ("76.789.012-3", "Asesorías Contables SpA"),
    ("76.890.123-4", "Transportes Norte Sur Ltda"),
    ("76.901.234-5", "Alimentos del Sur SA"),
    ("76.012.345-6", "Construcciones Patagonia SpA"),
    ("77.111.222-3", "Farmacia Popular Ltda"),
    ("77.222.333-4", "Servicios Mineros SA"),
    ("77.333.444-5", "Agrícola Valle Verde SpA"),
    ("77.444.555-6", "Editorial Nuevo Mundo Ltda"),
    ("77.555.666-7", "Consultoría Estratégica SA"),
]

CATEGORIAS = [
    "Pago Proveedores",
    "Pago Remuneraciones",
    "Pago Servicios Básicos",
    "Pago Honorarios",
    "Transferencia Interna",
    "Pago Arriendo",
    "Compra Insumos",
    "Pago Impuestos",
    "Devolución Cliente",
    "Abono Préstamo",
]

SUCURSALES = ["Santiago Centro", "Providencia", "Las Condes", "Maipú", "Valparaíso",
              "Concepción", "Temuco", "Antofagasta"]

CANALES = ["Banca Web", "Sucursal", "APP Móvil", "TEF", "Caja"]


def _random_datetime(start: datetime, end: datetime) -> datetime:
    """Genera un datetime aleatorio entre start y end en horario laboral."""
    delta = (end - start).total_seconds()
    random_seconds = random.randint(0, int(delta))
    dt = start + timedelta(seconds=random_seconds)
    # Ajustar a horario laboral (lunes-viernes, 8:00-18:00)
    while dt.weekday() >= 5:  # fin de semana
        dt += timedelta(days=1)
    dt = dt.replace(hour=random.randint(8, 17), minute=random.randint(0, 59),
                    second=random.randint(0, 59))
    return dt


def _generate_normal_amount(categoria: str) -> float:
    """Genera un monto realista según la categoría de transacción."""
    ranges = {
        "Pago Proveedores": (50_000, 5_000_000),
        "Pago Remuneraciones": (400_000, 3_500_000),
        "Pago Servicios Básicos": (20_000, 500_000),
        "Pago Honorarios": (100_000, 2_000_000),
        "Transferencia Interna": (100_000, 10_000_000),
        "Pago Arriendo": (300_000, 2_500_000),
        "Compra Insumos": (10_000, 800_000),
        "Pago Impuestos": (50_000, 8_000_000),
        "Devolución Cliente": (5_000, 500_000),
        "Abono Préstamo": (200_000, 5_000_000),
    }
    low, high = ranges.get(categoria, (10_000, 1_000_000))
    # Distribución log-normal para montos más realistas
    mean = (low + high) / 2
    amount = random.gauss(mean, mean * 0.3)
    return max(low * 0.5, min(high * 1.5, round(amount, 0)))


def _generate_normal_transactions(n: int, start_date: datetime,
                                  end_date: datetime) -> list[dict[str, Any]]:
    """Genera n transacciones normales (sin anomalías)."""
    transactions = []
    for i in range(n):
        categoria = random.choice(CATEGORIAS)
        cuenta_code, cuenta_name = random.choice(CUENTAS_ORIGEN)
        dest_rut, dest_name = random.choice(DESTINATARIOS)
        sucursal = random.choice(SUCURSALES)
        canal = random.choice(CANALES)
        dt = _random_datetime(start_date, end_date)
        monto = _generate_normal_amount(categoria)

        transactions.append({
            "id_transaccion": f"TXN-{i+1:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": cuenta_code,
            "cuenta_nombre": cuenta_name,
            "destinatario_rut": dest_rut,
            "destinatario_nombre": dest_name,
            "categoria": categoria,
            "monto": monto,
            "sucursal": sucursal,
            "canal": canal,
            "descripcion": f"{categoria} — {dest_name}",
            "es_anomalia": False,
            "tipo_anomalia": "normal",
        })
    return transactions


def _inject_high_amount_anomalies(transactions: list, start_id: int,
                                  start_date: datetime, end_date: datetime,
                                  n: int = 80) -> list[dict]:
    """Inyecta transacciones con montos extremadamente altos."""
    anomalies = []
    for i in range(n):
        base = random.choice(transactions)
        dt = _random_datetime(start_date, end_date)
        # Montos 5x-20x por encima del rango normal
        monto = random.uniform(15_000_000, 80_000_000)
        anomalies.append({
            "id_transaccion": f"TXN-{start_id + i:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": base["cuenta_origen"],
            "cuenta_nombre": base["cuenta_nombre"],
            "destinatario_rut": base["destinatario_rut"],
            "destinatario_nombre": base["destinatario_nombre"],
            "categoria": base["categoria"],
            "monto": round(monto, 0),
            "sucursal": base["sucursal"],
            "canal": base["canal"],
            "descripcion": f"{base['categoria']} — {base['destinatario_nombre']}",
            "es_anomalia": True,
            "tipo_anomalia": "monto_atipico_alto",
        })
    return anomalies


def _inject_negative_amount_anomalies(transactions: list, start_id: int,
                                      start_date: datetime, end_date: datetime,
                                      n: int = 30) -> list[dict]:
    """Inyecta transacciones con montos negativos sospechosos."""
    anomalies = []
    for i in range(n):
        base = random.choice(transactions)
        dt = _random_datetime(start_date, end_date)
        monto = -random.uniform(500_000, 10_000_000)
        anomalies.append({
            "id_transaccion": f"TXN-{start_id + i:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": base["cuenta_origen"],
            "cuenta_nombre": base["cuenta_nombre"],
            "destinatario_rut": base["destinatario_rut"],
            "destinatario_nombre": base["destinatario_nombre"],
            "categoria": "Devolución Cliente",
            "monto": round(monto, 0),
            "sucursal": base["sucursal"],
            "canal": base["canal"],
            "descripcion": f"Reversión sospechosa — {base['destinatario_nombre']}",
            "es_anomalia": True,
            "tipo_anomalia": "monto_negativo",
        })
    return anomalies


def _inject_off_hours_anomalies(start_id: int, start_date: datetime,
                                end_date: datetime, n: int = 60) -> list[dict]:
    """Inyecta transacciones en horarios fuera de patrón (madrugada/fin de semana)."""
    anomalies = []
    for i in range(n):
        categoria = random.choice(CATEGORIAS)
        cuenta_code, cuenta_name = random.choice(CUENTAS_ORIGEN)
        dest_rut, dest_name = random.choice(DESTINATARIOS)
        sucursal = random.choice(SUCURSALES)

        # Horario fuera de patrón: 00:00 - 05:59 o fin de semana
        dt = _random_datetime(start_date, end_date)
        if i % 2 == 0:
            # Madrugada entre semana
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
            dt = dt.replace(hour=random.randint(0, 4), minute=random.randint(0, 59))
        else:
            # Fin de semana
            while dt.weekday() < 5:
                dt += timedelta(days=1)
            dt = dt.replace(hour=random.randint(0, 23), minute=random.randint(0, 59))

        monto = _generate_normal_amount(categoria)
        anomalies.append({
            "id_transaccion": f"TXN-{start_id + i:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": cuenta_code,
            "cuenta_nombre": cuenta_name,
            "destinatario_rut": dest_rut,
            "destinatario_nombre": dest_name,
            "categoria": categoria,
            "monto": round(monto, 0),
            "sucursal": sucursal,
            "canal": random.choice(["Banca Web", "APP Móvil"]),
            "descripcion": f"{categoria} — {dest_name}",
            "es_anomalia": True,
            "tipo_anomalia": "horario_inusual",
        })
    return anomalies


def _inject_burst_anomalies(start_id: int, start_date: datetime,
                            end_date: datetime, n_bursts: int = 15,
                            txns_per_burst: int = 4) -> list[dict]:
    """Inyecta ráfagas de transacciones en pocos minutos (frecuencia inusual)."""
    anomalies = []
    idx = 0
    for _ in range(n_bursts):
        dest_rut, dest_name = random.choice(DESTINATARIOS)
        cuenta_code, cuenta_name = random.choice(CUENTAS_ORIGEN)
        categoria = random.choice(CATEGORIAS)
        base_dt = _random_datetime(start_date, end_date)
        sucursal = random.choice(SUCURSALES)

        for j in range(txns_per_burst):
            dt = base_dt + timedelta(seconds=random.randint(10, 180))
            monto = _generate_normal_amount(categoria) * random.uniform(0.8, 1.2)
            anomalies.append({
                "id_transaccion": f"TXN-{start_id + idx:06d}",
                "fecha": dt.strftime("%Y-%m-%d"),
                "hora": dt.strftime("%H:%M:%S"),
                "dia_semana": dt.strftime("%A"),
                "cuenta_origen": cuenta_code,
                "cuenta_nombre": cuenta_name,
                "destinatario_rut": dest_rut,
                "destinatario_nombre": dest_name,
                "categoria": categoria,
                "monto": round(monto, 0),
                "sucursal": sucursal,
                "canal": random.choice(["Banca Web", "APP Móvil"]),
                "descripcion": f"{categoria} — {dest_name} (ráfaga)",
                "es_anomalia": True,
                "tipo_anomalia": "frecuencia_inusual",
            })
            idx += 1
    return anomalies


def _inject_duplicate_anomalies(transactions: list, start_id: int,
                                n: int = 40) -> list[dict]:
    """Inyecta duplicados sospechosos (mismo monto y destinatario, segundos después)."""
    anomalies = []
    candidates = random.sample(transactions, min(n, len(transactions)))
    for i, base in enumerate(candidates):
        dt = datetime.strptime(f"{base['fecha']} {base['hora']}", "%Y-%m-%d %H:%M:%S")
        dt += timedelta(seconds=random.randint(5, 120))
        anomalies.append({
            "id_transaccion": f"TXN-{start_id + i:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": base["cuenta_origen"],
            "cuenta_nombre": base["cuenta_nombre"],
            "destinatario_rut": base["destinatario_rut"],
            "destinatario_nombre": base["destinatario_nombre"],
            "categoria": base["categoria"],
            "monto": base["monto"],  # Mismo monto exacto
            "sucursal": base["sucursal"],
            "canal": base["canal"],
            "descripcion": f"{base['categoria']} — {base['destinatario_nombre']} (duplicado)",
            "es_anomalia": True,
            "tipo_anomalia": "duplicado_sospechoso",
        })
    return anomalies


def _inject_new_account_anomalies(start_id: int, start_date: datetime,
                                  end_date: datetime, n: int = 40) -> list[dict]:
    """Inyecta transacciones altas desde cuentas/destinatarios nunca antes vistos."""
    new_destinations = [
        ("99.111.000-1", "Empresa Fantasma Alpha SpA"),
        ("99.222.000-2", "Offshore Holdings Beta Ltda"),
        ("99.333.000-3", "Inversiones Gamma SA"),
        ("99.444.000-4", "Comercial Delta Express SpA"),
        ("99.555.000-5", "Trading Epsilon Internacional Ltda"),
        ("99.666.000-6", "Servicios Zeta Consultoría SA"),
        ("99.777.000-7", "Importadora Eta Global SpA"),
        ("99.888.000-8", "Distribuidora Theta Ltda"),
    ]
    anomalies = []
    for i in range(n):
        dest_rut, dest_name = random.choice(new_destinations)
        cuenta_code, cuenta_name = random.choice(CUENTAS_ORIGEN)
        dt = _random_datetime(start_date, end_date)
        monto = random.uniform(8_000_000, 50_000_000)
        anomalies.append({
            "id_transaccion": f"TXN-{start_id + i:06d}",
            "fecha": dt.strftime("%Y-%m-%d"),
            "hora": dt.strftime("%H:%M:%S"),
            "dia_semana": dt.strftime("%A"),
            "cuenta_origen": cuenta_code,
            "cuenta_nombre": cuenta_name,
            "destinatario_rut": dest_rut,
            "destinatario_nombre": dest_name,
            "categoria": random.choice(["Transferencia Interna", "Pago Proveedores"]),
            "monto": round(monto, 0),
            "sucursal": random.choice(SUCURSALES),
            "canal": random.choice(["Banca Web", "APP Móvil"]),
            "descripcion": f"Primera transacción — {dest_name}",
            "es_anomalia": True,
            "tipo_anomalia": "cuenta_nueva_monto_alto",
        })
    return anomalies


def generate_transactions(output_path: str | None = None,
                          n_normal: int = 4700) -> str:
    """
    Genera el dataset completo de transacciones con anomalías inyectadas.

    Retorna la ruta del archivo CSV generado.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "transacciones.csv")

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    # Generar transacciones normales
    print(f"  Generando {n_normal} transacciones normales...")
    normal = _generate_normal_transactions(n_normal, start_date, end_date)

    # Inyectar anomalías
    next_id = n_normal + 1
    print("  Inyectando anomalías:")

    high_amount = _inject_high_amount_anomalies(normal, next_id, start_date, end_date, n=80)
    print(f"    - Montos atípicos altos     : {len(high_amount)}")
    next_id += len(high_amount)

    negative = _inject_negative_amount_anomalies(normal, next_id, start_date, end_date, n=30)
    print(f"    - Montos negativos          : {len(negative)}")
    next_id += len(negative)

    off_hours = _inject_off_hours_anomalies(next_id, start_date, end_date, n=60)
    print(f"    - Horario inusual           : {len(off_hours)}")
    next_id += len(off_hours)

    bursts = _inject_burst_anomalies(next_id, start_date, end_date, n_bursts=15, txns_per_burst=4)
    print(f"    - Frecuencia inusual        : {len(bursts)}")
    next_id += len(bursts)

    duplicates = _inject_duplicate_anomalies(normal, next_id, n=40)
    print(f"    - Duplicados sospechosos    : {len(duplicates)}")
    next_id += len(duplicates)

    new_account = _inject_new_account_anomalies(next_id, start_date, end_date, n=40)
    print(f"    - Cuenta nueva + monto alto : {len(new_account)}")

    # Combinar y mezclar
    all_transactions = normal + high_amount + negative + off_hours + bursts + duplicates + new_account
    random.shuffle(all_transactions)

    # Reasignar IDs secuenciales después de mezclar
    for i, txn in enumerate(all_transactions):
        txn["id_transaccion"] = f"TXN-{i+1:06d}"

    # Escribir CSV
    fieldnames = [
        "id_transaccion", "fecha", "hora", "dia_semana", "cuenta_origen",
        "cuenta_nombre", "destinatario_rut", "destinatario_nombre", "categoria",
        "monto", "sucursal", "canal", "descripcion", "es_anomalia", "tipo_anomalia",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_transactions)

    total = len(all_transactions)
    anomalies_count = sum(1 for t in all_transactions if t["es_anomalia"])
    print(f"\n  Total transacciones  : {total:,}")
    print(f"  Normales             : {total - anomalies_count:,}")
    print(f"  Anomalías inyectadas : {anomalies_count:,} ({anomalies_count/total*100:.1f}%)")
    print(f"  Archivo generado     : {output_path}")

    return output_path
