import json
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
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
    dc_ir = Column(Float, nullable=True)
    ac_ir = Column(Float, nullable=True)
    transmitancia_dc = Column(Float, nullable=True)
    transmitancia_ac = Column(Float, nullable=True)

    # Features calculadas
    ratio_ir_trans = Column(Float, nullable=True)
    pulsatile_index = Column(Float, nullable=True)
    ir_ratio = Column(Float, nullable=True)

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


def get_db_session():
    return SessionLocal()


Base.metadata.create_all(bind=engine)


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
        
        new_reading = MLTrainingData(
            bpm=data.get("bpm"),
            dc_ir=data.get("dc_ir"),
            ac_ir=data.get("ac_ir"),
            transmitancia_dc=data.get("transmitancia_dc"),
            transmitancia_ac=data.get("transmitancia_ac"),
        )

        # Calcular features
        if new_reading.transmitancia_ac and new_reading.ac_ir and new_reading.ac_ir != 0:
            new_reading.ratio_ir_trans = new_reading.transmitancia_ac / new_reading.ac_ir

        if new_reading.transmitancia_ac and new_reading.transmitancia_dc and new_reading.transmitancia_dc != 0:
            new_reading.pulsatile_index = new_reading.transmitancia_ac / new_reading.transmitancia_dc

        if new_reading.ac_ir and new_reading.dc_ir and new_reading.dc_ir != 0:
            new_reading.ir_ratio = new_reading.ac_ir / new_reading.dc_ir

        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)

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

        new_record = MLTrainingData(
            bpm=data.get("bpm"),
            dc_ir=data.get("dc_ir"),
            ac_ir=data.get("ac_ir"),
            transmitancia_dc=data.get("transmitancia_dc"),
            transmitancia_ac=data.get("transmitancia_ac"),
            idade=data.get("idade"),
            peso=data.get("peso"),
            altura=data.get("altura"),
            sexo=data.get("sexo"),
            ultima_refeicao_horas=data.get("ultima_refeicao_horas"),
            atividade_recente=data.get("atividade_recente"),
            glicose_real=data.get("glicose_real"),
            glicose_estimada=data.get("glicose_estimada"),
        )

        # Calcular features
        if new_record.transmitancia_ac and new_record.ac_ir and new_record.ac_ir != 0:
            new_record.ratio_ir_trans = new_record.transmitancia_ac / new_record.ac_ir

        if new_record.transmitancia_ac and new_record.transmitancia_dc and new_record.transmitancia_dc != 0:
            new_record.pulsatile_index = new_record.transmitancia_ac / new_record.transmitancia_dc

        if new_record.ac_ir and new_record.dc_ir and new_record.dc_ir != 0:
            new_record.ir_ratio = new_record.ac_ir / new_record.dc_ir

        if new_record.peso and new_record.altura and new_record.altura != 0:
            new_record.imc = new_record.peso / (new_record.altura ** 2)

        # Calcular erros
        if new_record.glicose_real and new_record.glicose_estimada:
            new_record.erro_absoluto = abs(new_record.glicose_real - new_record.glicose_estimada)
            if new_record.glicose_real != 0:
                new_record.erro_percentual = (new_record.erro_absoluto / new_record.glicose_real) * 100

        db.add(new_record)
        db.commit()
        db.refresh(new_record)

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
                "dc_ir": r.dc_ir,
                "ac_ir": r.ac_ir,
                "transmitancia_dc": r.transmitancia_dc,
                "transmitancia_ac": r.transmitancia_ac,
                "ratio_ir_trans": r.ratio_ir_trans,
                "pulsatile_index": r.pulsatile_index,
                "ir_ratio": r.ir_ratio,
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

        # Atualizar campos
        if "glicose_real" in data:
            record.glicose_real = data["glicose_real"]
        if "idade" in data:
            record.idade = data["idade"]
        if "peso" in data:
            record.peso = data["peso"]
        if "altura" in data:
            record.altura = data["altura"]
        if "sexo" in data:
            record.sexo = data["sexo"]
        if "ultima_refeicao_horas" in data:
            record.ultima_refeicao_horas = data["ultima_refeicao_horas"]
        if "atividade_recente" in data:
            record.atividade_recente = data["atividade_recente"]

        # Recalcular IMC
        if record.peso and record.altura and record.altura != 0:
            record.imc = record.peso / (record.altura ** 2)

        # Recalcular erros
        if record.glicose_real and record.glicose_estimada:
            record.erro_absoluto = abs(record.glicose_real - record.glicose_estimada)
            if record.glicose_real != 0:
                record.erro_percentual = (record.erro_absoluto / record.glicose_real) * 100

        db.commit()
        db.refresh(record)

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
