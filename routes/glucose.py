import json
import math
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "health_database.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

glucose_routes = Blueprint("glucose_routes", __name__)


class GlucoseReading(Base):
    """Modelo local para leituras de glicose."""
    __tablename__ = "glucose_data"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    glucose_mg_dl = Column(Float, nullable=True)
    spectral_transmittance_data = Column(Text, nullable=True)
    box_temperature_celsius = Column(Float, nullable=True)
    measurement_phase = Column(String(80), nullable=True)
    real_concentration = Column(Float, nullable=True)
    volunteer_notes = Column(Text, nullable=True)


class MLTrainingData(Base):
    """Modelo para dados de treinamento de IA de glicose."""
    __tablename__ = "ml_training_data"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Sensores
    bpm = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    dc_ir = Column(Float, nullable=True)
    ac_ir = Column(Float, nullable=True)
    ir_max30102 = Column(Float, nullable=True)
    red_max30102 = Column(Float, nullable=True)
    transmitancia_dc = Column(Float, nullable=True)
    transmitancia_ac = Column(Float, nullable=True)
    bpw34_raw = Column(Float, nullable=True)
    bpw34_voltage = Column(Float, nullable=True)
    bpw34_current = Column(Float, nullable=True)
    bpw34_ac = Column(Float, nullable=True)
    bpw34_dc = Column(Float, nullable=True)
    bpw34_rms = Column(Float, nullable=True)
    bpw34_peak = Column(Float, nullable=True)
    bpw34_mean = Column(Float, nullable=True)
    ir_940_intensity = Column(Float, nullable=True)
    ir_940_transmittance = Column(Float, nullable=True)
    red_660 = Column(Float, nullable=True)
    temperatura = Column(Float, nullable=True)

    # Features calculadas
    transmittance = Column(Float, nullable=True)
    absorbance = Column(Float, nullable=True)
    ratio_ir_trans = Column(Float, nullable=True)
    pulsatile_index = Column(Float, nullable=True)
    ir_ratio = Column(Float, nullable=True)
    ratio_ir_bpw34 = Column(Float, nullable=True)

    # Dados do usuário
    idade = Column(Integer, nullable=True)
    peso = Column(Float, nullable=True)
    altura = Column(Float, nullable=True)
    imc = Column(Float, nullable=True)
    sexo = Column(String(10), nullable=True)

    # Contexto
    ultima_refeicao_horas = Column(Float, nullable=True)
    atividade_recente = Column(Integer, nullable=True)  # 0=nenhuma, 1=leve, 2=intensa

    # Glicose
    glicose_real = Column(Float, nullable=True)
    glicose_estimada = Column(Float, nullable=True)
    erro_absoluto = Column(Float, nullable=True)
    erro_percentual = Column(Float, nullable=True)


class OpticalRawData(Base):
    """Tabela bruta para preservar leituras opticas do hardware."""
    __tablename__ = "optical_raw_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    ml_training_data_id = Column(Integer, nullable=True)

    bpw34_raw = Column(Float, nullable=True)
    bpw34_voltage = Column(Float, nullable=True)
    ir_940 = Column(Float, nullable=True)
    red_660 = Column(Float, nullable=True)
    bpm = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    glicose_real = Column(Float, nullable=True)


def get_db_session():
    return SessionLocal()


Base.metadata.create_all(bind=engine)


ML_TRAINING_EXTRA_COLUMNS = {
    "spo2": "FLOAT",
    "ir_max30102": "FLOAT",
    "red_max30102": "FLOAT",
    "bpw34_raw": "FLOAT",
    "bpw34_voltage": "FLOAT",
    "bpw34_current": "FLOAT",
    "bpw34_ac": "FLOAT",
    "bpw34_dc": "FLOAT",
    "bpw34_rms": "FLOAT",
    "bpw34_peak": "FLOAT",
    "bpw34_mean": "FLOAT",
    "ir_940_intensity": "FLOAT",
    "ir_940_transmittance": "FLOAT",
    "red_660": "FLOAT",
    "temperatura": "FLOAT",
    "transmittance": "FLOAT",
    "absorbance": "FLOAT",
    "ratio_ir_bpw34": "FLOAT",
}


def migrate_sqlite_schema() -> None:
    """Adiciona colunas novas em bancos SQLite existentes sem apagar dados."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    if "ml_training_data" in existing_tables:
        existing_columns = {col["name"] for col in inspector.get_columns("ml_training_data")}
        with engine.begin() as conn:
            for column_name, column_type in ML_TRAINING_EXTRA_COLUMNS.items():
                if column_name not in existing_columns:
                    conn.execute(text(f"ALTER TABLE ml_training_data ADD COLUMN {column_name} {column_type}"))


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_derived_features(data: dict) -> dict:
    """Calcula features opticas derivadas preservando compatibilidade com nomes antigos."""
    result = dict(data)

    received_light = safe_float(
        result.get("ir_940_transmittance")
        or result.get("bpw34_voltage")
        or result.get("bpw34_dc")
        or result.get("transmitancia_dc")
    )
    emitted_light = safe_float(result.get("ir_940_intensity") or result.get("ir_max30102") or result.get("dc_ir"))
    if received_light is not None and emitted_light and emitted_light > 0:
        transmittance = received_light / emitted_light
        result["transmittance"] = transmittance
        if transmittance > 0:
            result["absorbance"] = -math.log10(transmittance)

    ac_component = safe_float(result.get("bpw34_ac") or result.get("transmitancia_ac"))
    dc_component = safe_float(result.get("bpw34_dc") or result.get("transmitancia_dc"))
    if ac_component is not None and dc_component:
        result["pulsatile_index"] = ac_component / dc_component

    transmitancia_ac = safe_float(result.get("transmitancia_ac"))
    ac_ir = safe_float(result.get("ac_ir"))
    if transmitancia_ac is not None and ac_ir:
        result["ratio_ir_trans"] = transmitancia_ac / ac_ir

    dc_ir = safe_float(result.get("dc_ir"))
    if ac_ir is not None and dc_ir:
        result["ir_ratio"] = ac_ir / dc_ir

    ir_940 = safe_float(result.get("ir_940_intensity"))
    bpw34_voltage = safe_float(result.get("bpw34_voltage"))
    if ir_940 is not None and bpw34_voltage:
        result["ratio_ir_bpw34"] = ir_940 / bpw34_voltage

    return result


def apply_payload_to_record(record: MLTrainingData, data: dict) -> MLTrainingData:
    allowed_fields = [
        "bpm", "spo2", "dc_ir", "ac_ir", "ir_max30102", "red_max30102",
        "transmitancia_dc", "transmitancia_ac", "bpw34_raw", "bpw34_voltage",
        "bpw34_current", "bpw34_ac", "bpw34_dc", "bpw34_rms", "bpw34_peak",
        "bpw34_mean", "ir_940_intensity", "ir_940_transmittance", "red_660",
        "temperatura", "transmittance", "absorbance", "ratio_ir_trans",
        "pulsatile_index", "ir_ratio", "ratio_ir_bpw34", "idade", "peso",
        "altura", "sexo", "ultima_refeicao_horas", "atividade_recente",
        "glicose_real", "glicose_estimada",
    ]

    data = compute_derived_features(data)
    for field in allowed_fields:
        if field in data:
            value = data[field]
            setattr(record, field, value if field == "sexo" else safe_float(value))

    if record.peso and record.altura:
        record.imc = record.peso / (record.altura ** 2)

    if record.glicose_real and record.glicose_estimada:
        record.erro_absoluto = abs(record.glicose_real - record.glicose_estimada)
        if record.glicose_real != 0:
            record.erro_percentual = (record.erro_absoluto / record.glicose_real) * 100

    return record


def create_optical_raw_row(record: MLTrainingData) -> OpticalRawData:
    return OpticalRawData(
        ml_training_data_id=record.id,
        bpw34_raw=record.bpw34_raw,
        bpw34_voltage=record.bpw34_voltage,
        ir_940=record.ir_940_intensity,
        red_660=record.red_660,
        bpm=record.bpm,
        spo2=record.spo2,
        glicose_real=record.glicose_real,
    )


migrate_sqlite_schema()


# Rota GET existente para ler os dados
@glucose_routes.route("/glucose", methods=["GET"])
def get_glucose():
    db = get_db_session()
    try:
        readings = db.query(GlucoseReading).order_by(GlucoseReading.created_at.desc()).all()
        result = [
            {
                "id": reading.id,
                "created_at": reading.created_at.isoformat(),
                "glucose_mg_dl": reading.glucose_mg_dl,
                "spectral_transmittance_data": json.loads(reading.spectral_transmittance_data)
                if reading.spectral_transmittance_data else None,
                "box_temperature_celsius": reading.box_temperature_celsius,
                "measurement_phase": reading.measurement_phase,
                "real_concentration": reading.real_concentration,
                "volunteer_notes": reading.volunteer_notes,
            }
            for reading in readings
        ]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# Nova Rota POST para receber dados do Raspberry Pi
@glucose_routes.route("/glucose", methods=["POST"])
def add_glucose_reading():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Payload JSON inválido"}), 400

        spectral_transmittance_data = data.get("spectral_transmittance_data")
        if isinstance(spectral_transmittance_data, dict):
            spectral_transmittance_data = json.dumps(spectral_transmittance_data, ensure_ascii=False)

        new_reading = GlucoseReading(
            glucose_mg_dl=data.get("glucose_mg_dl"),
            spectral_transmittance_data=spectral_transmittance_data,
            box_temperature_celsius=data.get("box_temperature_celsius"),
            measurement_phase=data.get("measurement_phase"),
            real_concentration=data.get("real_concentration"),
            volunteer_notes=data.get("volunteer_notes", ""),
        )

        db = get_db_session()
        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)
        return jsonify({
            "message": "Leitura salva com sucesso!",
            "data": {
                "id": new_reading.id,
                "created_at": new_reading.created_at.isoformat(),
            }
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'db' in locals():
            db.close()

# ============================================================================
# NOVOS ENDPOINTS PARA TREINAMENTO DE IA
# ============================================================================

@glucose_routes.route("/sensor-reading", methods=["POST"])
def add_sensor_reading():
    """Registra uma leitura de sensores (BPM, DC_IR, AC_IR, Transmitancia)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Payload JSON inválido"}), 400

        db = get_db_session()
        
        new_reading = apply_payload_to_record(MLTrainingData(), data)

        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)
        db.add(create_optical_raw_row(new_reading))
        db.commit()

        return jsonify({
            "message": "Leitura de sensor salva com sucesso!",
            "data": {
                "id": new_reading.id,
                "created_at": new_reading.created_at.isoformat(),
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'db' in locals():
            db.close()


@glucose_routes.route("/training-data", methods=["POST"])
def add_training_data():
    """Registra dados de treinamento com glicose real medida."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Payload JSON inválido"}), 400

        db = get_db_session()

        new_record = apply_payload_to_record(MLTrainingData(), data)

        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        db.add(create_optical_raw_row(new_record))
        db.commit()

        return jsonify({
            "message": "Dados de treinamento salvos com sucesso!",
            "data": {
                "id": new_record.id,
                "created_at": new_record.created_at.isoformat(),
                "imc": new_record.imc,
                "erro_absoluto": new_record.erro_absoluto,
                "erro_percentual": new_record.erro_percentual,
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'db' in locals():
            db.close()


@glucose_routes.route("/training-data", methods=["GET"])
def get_training_data():
    """Retorna todos os dados de treinamento."""
    try:
        db = get_db_session()
        records = db.query(MLTrainingData).order_by(MLTrainingData.created_at.desc()).all()

        result = [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "bpm": r.bpm,
                "spo2": r.spo2,
                "dc_ir": r.dc_ir,
                "ac_ir": r.ac_ir,
                "ir_max30102": r.ir_max30102,
                "red_max30102": r.red_max30102,
                "transmitancia_dc": r.transmitancia_dc,
                "transmitancia_ac": r.transmitancia_ac,
                "bpw34_raw": r.bpw34_raw,
                "bpw34_voltage": r.bpw34_voltage,
                "bpw34_current": r.bpw34_current,
                "bpw34_ac": r.bpw34_ac,
                "bpw34_dc": r.bpw34_dc,
                "bpw34_rms": r.bpw34_rms,
                "bpw34_peak": r.bpw34_peak,
                "bpw34_mean": r.bpw34_mean,
                "ir_940_intensity": r.ir_940_intensity,
                "ir_940_transmittance": r.ir_940_transmittance,
                "red_660": r.red_660,
                "temperatura": r.temperatura,
                "transmittance": r.transmittance,
                "absorbance": r.absorbance,
                "ratio_ir_trans": r.ratio_ir_trans,
                "pulsatile_index": r.pulsatile_index,
                "ir_ratio": r.ir_ratio,
                "ratio_ir_bpw34": r.ratio_ir_bpw34,
                "idade": r.idade,
                "peso": r.peso,
                "altura": r.altura,
                "imc": r.imc,
                "sexo": r.sexo,
                "ultima_refeicao_horas": r.ultima_refeicao_horas,
                "atividade_recente": r.atividade_recente,
                "glicose_real": r.glicose_real,
                "glicose_estimada": r.glicose_estimada,
                "erro_absoluto": r.erro_absoluto,
                "erro_percentual": r.erro_percentual,
            }
            for r in records
        ]

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@glucose_routes.route("/training-data/<int:record_id>", methods=["PUT"])
def update_training_data(record_id):
    """Atualiza um registro de treinamento (ex: adicionar glicose real)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Payload JSON inválido"}), 400

        db = get_db_session()
        record = db.query(MLTrainingData).filter(MLTrainingData.id == record_id).first()

        if not record:
            return jsonify({"error": "Registro não encontrado"}), 404

        apply_payload_to_record(record, data)

        db.commit()
        db.refresh(record)
        raw_record = db.query(OpticalRawData).filter(
            OpticalRawData.ml_training_data_id == record.id
        ).first()
        if raw_record:
            raw_record.glicose_real = record.glicose_real
        else:
            db.add(create_optical_raw_row(record))
        db.commit()

        return jsonify({
            "message": "Registro atualizado com sucesso!",
            "data": {
                "id": record.id,
                "erro_absoluto": record.erro_absoluto,
                "erro_percentual": record.erro_percentual,
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'db' in locals():
            db.close()
