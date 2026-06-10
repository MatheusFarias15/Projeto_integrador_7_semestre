"""
Secao de treinamento de IA para estimativa de glicose via hardware optico.

Pode ser renderizada dentro do dashboard principal com render_ai_training_section().
"""

import json
import math
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "health_database.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
MODELS_DIR = BASE_DIR / "models"


def get_db_session():
    """Retorna uma sessao SQLAlchemy e os modelos usados pela tela."""
    from routes.glucose import MLTrainingData, OpticalRawData

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal(), MLTrainingData, OpticalRawData


def format_value(value, decimals=2, suffix=""):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def load_latest_hardware_artifacts():
    """Carrega modelo, scaler e lista de features do treinamento do hardware."""
    model_path = MODELS_DIR / "hardware_glucose_model_latest.pkl"
    scaler_path = MODELS_DIR / "hardware_glucose_scaler_latest.pkl"
    features_path = MODELS_DIR / "hardware_feature_names_latest.json"

    if not model_path.exists():
        candidates = sorted(MODELS_DIR.glob("glucose_model_*.pkl"), reverse=True)
        model_path = candidates[0] if candidates else model_path

    if not scaler_path.exists():
        candidates = sorted(MODELS_DIR.glob("glucose_scaler_*.pkl"), reverse=True)
        scaler_path = candidates[0] if candidates else scaler_path

    feature_names = None
    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        report = load_latest_report(show_warning=False)
        feature_names = report.get("data_stats", {}).get("features") if report else None

    if not model_path.exists() or not scaler_path.exists() or not feature_names:
        return None, None, None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler, feature_names
    except Exception as exc:
        st.warning(f"Erro ao carregar modelo do hardware: {exc}")
        return None, None, None


def record_to_feature_dict(record) -> dict:
    return {
        column.name: getattr(record, column.name)
        for column in record.__table__.columns
        if column.name not in {"id", "created_at"}
    }


def predict_hardware_glucose(record) -> float | None:
    model, scaler, feature_names = load_latest_hardware_artifacts()
    if model is None or scaler is None or not feature_names:
        return None

    data = record_to_feature_dict(record)
    row = {feature: data.get(feature, 0) for feature in feature_names}
    df_input = pd.DataFrame([row]).fillna(0)

    try:
        scaled = scaler.transform(df_input[feature_names])
        prediction = float(model.predict(scaled)[0])
        return max(40.0, min(450.0, prediction))
    except Exception as exc:
        st.warning(f"Nao foi possivel estimar glicose com o modelo do hardware: {exc}")
        return None


def update_training_record(record_id: int, payload: dict) -> None:
    from routes.glucose import apply_payload_to_record

    db, MLTrainingData, OpticalRawData = get_db_session()
    try:
        record = db.query(MLTrainingData).filter(MLTrainingData.id == record_id).first()
        if record is None:
            raise ValueError("Registro nao encontrado")

        apply_payload_to_record(record, payload)
        db.commit()
        db.refresh(record)

        raw_record = db.query(OpticalRawData).filter(
            OpticalRawData.ml_training_data_id == record.id
        ).first()
        if raw_record:
            raw_record.glicose_real = record.glicose_real
        else:
            db.add(OpticalRawData(
                ml_training_data_id=record.id,
                bpw34_raw=record.bpw34_raw,
                bpw34_voltage=record.bpw34_voltage,
                ir_940=record.ir_940_intensity,
                red_660=record.red_660,
                bpm=record.bpm,
                spo2=record.spo2,
                glicose_real=record.glicose_real,
            ))
        db.commit()
    finally:
        db.close()


def get_latest_record(pending_only=False):
    db, MLTrainingData, _ = get_db_session()
    try:
        query = db.query(MLTrainingData)
        if pending_only:
            query = query.filter(MLTrainingData.glicose_real == None)  # noqa: E711
        return query.order_by(MLTrainingData.created_at.desc()).first()
    finally:
        db.close()


def get_training_dataframe(limit: int | None = None) -> pd.DataFrame:
    db, MLTrainingData, _ = get_db_session()
    try:
        query = db.query(MLTrainingData).order_by(MLTrainingData.created_at.desc())
        if limit:
            query = query.limit(limit)
        records = query.all()
        rows = []
        for r in records:
            rows.append({
                "id": r.id,
                "created_at": r.created_at,
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
                "glicose_estimada": r.glicose_estimada,
                "glicose_real": r.glicose_real,
                "erro_absoluto": r.erro_absoluto,
                "erro_percentual": r.erro_percentual,
                "idade": r.idade,
                "peso": r.peso,
                "altura": r.altura,
                "imc": r.imc,
                "sexo": r.sexo,
                "ultima_refeicao_horas": r.ultima_refeicao_horas,
                "atividade_recente": r.atividade_recente,
            })
        return pd.DataFrame(rows)
    finally:
        db.close()


def get_optical_raw_dataframe(limit: int | None = None) -> pd.DataFrame:
    db, _, OpticalRawData = get_db_session()
    try:
        query = db.query(OpticalRawData).order_by(OpticalRawData.timestamp.desc())
        if limit:
            query = query.limit(limit)
        records = query.all()
        return pd.DataFrame([
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "ml_training_data_id": r.ml_training_data_id,
                "bpw34_raw": r.bpw34_raw,
                "bpw34_voltage": r.bpw34_voltage,
                "ir_940": r.ir_940,
                "red_660": r.red_660,
                "bpm": r.bpm,
                "spo2": r.spo2,
                "glicose_real": r.glicose_real,
            }
            for r in records
        ])
    finally:
        db.close()


def load_latest_report(show_warning=True):
    if not MODELS_DIR.exists():
        if show_warning:
            st.warning("Diretorio de modelos ainda nao existe.")
        return None

    latest_alias = MODELS_DIR / "hardware_training_report_latest.json"
    if latest_alias.exists():
        report_path = latest_alias
    else:
        reports = sorted(MODELS_DIR.glob("glucose_training_report_*.json"), reverse=True)
        report_path = reports[0] if reports else None

    if not report_path:
        if show_warning:
            st.warning("Nenhum relatorio de treinamento encontrado. Execute o treinamento primeiro.")
        return None

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_metric_grid(record):
    predicted = predict_hardware_glucose(record)
    if predicted is not None and record.glicose_estimada != predicted:
        update_training_record(record.id, {"glicose_estimada": predicted})

    st.caption(f"Registro #{record.id} capturado em {record.created_at.strftime('%d/%m/%Y %H:%M:%S')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BPM", format_value(record.bpm, 0))
    col2.metric("SpO2", format_value(record.spo2, 1, "%"))
    col3.metric("Temperatura", format_value(record.temperatura, 1, " C"))
    col4.metric("Glicose estimada", format_value(predicted or record.glicose_estimada, 1, " mg/dL"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BPW34 raw", format_value(record.bpw34_raw, 0))
    col2.metric("BPW34 voltage", format_value(record.bpw34_voltage, 4, " V"))
    col3.metric("BPW34 AC", format_value(record.bpw34_ac or record.transmitancia_ac, 4))
    col4.metric("BPW34 DC", format_value(record.bpw34_dc or record.transmitancia_dc, 4))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IR MAX30102", format_value(record.ir_max30102 or record.dc_ir, 0))
    col2.metric("Red MAX30102", format_value(record.red_max30102, 0))
    col3.metric("IR 940nm", format_value(record.ir_940_intensity, 4))
    col4.metric("Vermelho 660nm", format_value(record.red_660, 4))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transmitancia", format_value(record.transmittance, 6))
    col2.metric("Absorbancia", format_value(record.absorbance, 6))
    col3.metric("Pulsatile index", format_value(record.pulsatile_index, 6))
    col4.metric("Ratio IR/BPW34", format_value(record.ratio_ir_bpw34, 6))


def render_hardware_collection_tab():
    st.subheader("Coleta do Hardware")
    record = get_latest_record()

    if record is None:
        st.info("Nenhuma leitura encontrada. Execute app.py e serial_reader.py para receber dados do ESP32.")
        return

    render_metric_grid(record)

    st.divider()
    df_recent = get_training_dataframe(limit=20)
    if not df_recent.empty:
        st.write("Ultimas leituras recebidas")
        st.dataframe(df_recent, use_container_width=True, hide_index=True)


def render_validation_tab():
    st.subheader("Validacao da Glicose")
    record = get_latest_record(pending_only=True)

    if record is None:
        st.success("Nao ha registros pendentes de glicose real.")
        df_recent = get_training_dataframe(limit=10)
        if not df_recent.empty:
            st.dataframe(df_recent, use_container_width=True, hide_index=True)
        return

    st.info(f"Registro pendente #{record.id}")
    render_metric_grid(record)

    st.divider()
    with st.form("hardware_validation_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            glicose_real = st.number_input("Glicose real (mg/dL)", min_value=30.0, max_value=450.0, value=100.0, step=1.0)
            idade = st.number_input("Idade", min_value=0, max_value=120, value=int(record.idade or 30))
        with col2:
            peso = st.number_input("Peso (kg)", min_value=20.0, max_value=250.0, value=float(record.peso or 70.0), step=0.5)
            altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=float(record.altura or 1.70), step=0.01)
        with col3:
            sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"], index=0)
            ultima_refeicao = st.number_input("Ultima refeicao (horas)", min_value=0.0, max_value=24.0, value=float(record.ultima_refeicao_horas or 2.0), step=0.5)
            atividade = st.selectbox("Atividade recente", ["Nenhuma (0)", "Leve (1)", "Intensa (2)"])

        submitted = st.form_submit_button("Salvar validacao", use_container_width=True, type="primary")

    if submitted:
        atividade_valor = int(atividade.split("(")[1].rstrip(")"))
        update_training_record(record.id, {
            "glicose_real": glicose_real,
            "idade": idade,
            "peso": peso,
            "altura": altura,
            "sexo": sexo,
            "ultima_refeicao_horas": ultima_refeicao,
            "atividade_recente": atividade_valor,
        })
        st.success(f"Glicose real salva: {glicose_real:.0f} mg/dL")
        st.rerun()


def render_training_tab():
    st.subheader("Treinamento")
    df = get_training_dataframe()
    valid_count = int(df["glicose_real"].notna().sum()) if not df.empty else 0
    st.metric("Registros validados", valid_count)

    st.info(
        "O treinamento usa apenas registros com glicose real validada e calcula importancia por "
        "XGBoost/modelo, SHAP, Pearson, Spearman e Mutual Information."
    )

    if st.button("Iniciar treinamento do hardware", key="hardware_train", use_container_width=True, type="primary"):
        if valid_count < 10:
            st.error(f"Dados insuficientes: apenas {valid_count} registros validados. Colete pelo menos 10.")
            return

        with st.spinner("Treinando modelos do hardware..."):
            try:
                result = subprocess.run(
                    ["python", "train_glucose_model_cli.py"],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    st.success("Treinamento concluido com sucesso.")
                    st.code(result.stdout, language="text")
                else:
                    st.error("Erro durante o treinamento.")
                    st.code(result.stderr or result.stdout, language="text")
            except subprocess.TimeoutExpired:
                st.error("Treinamento excedeu o tempo limite.")
            except Exception as exc:
                st.error(f"Erro ao iniciar treinamento: {exc}")


def render_analysis_tab():
    st.subheader("Analise Academica")
    report = load_latest_report()
    if not report:
        return

    st.caption(f"Relatorio gerado em {report.get('timestamp', '-')}")
    models = report.get("models", {})
    if not models:
        st.warning("Relatorio sem dados de modelos.")
        return

    metrics_rows = []
    for model_name, model_info in models.items():
        metrics = model_info.get("metrics", {})
        metrics_rows.append({
            "Modelo": model_name,
            "R2": metrics.get("r2_score", 0),
            "RMSE": metrics.get("rmse", 0),
            "MAE": metrics.get("mae", 0),
            "MAPE": metrics.get("mape", 0),
        })

    df_metrics = pd.DataFrame(metrics_rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.bar_chart(df_metrics.set_index("Modelo")[["R2", "RMSE", "MAE"]])

    best_model = report.get("best_model")
    best_metrics = report.get("best_metrics", {})
    st.divider()
    st.subheader(f"Melhor modelo: {str(best_model).upper()}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R2", format_value(best_metrics.get("r2_score"), 4))
    col2.metric("RMSE", format_value(best_metrics.get("rmse"), 2, " mg/dL"))
    col3.metric("MAE", format_value(best_metrics.get("mae"), 2, " mg/dL"))
    col4.metric("MAPE", format_value(best_metrics.get("mape"), 4))

    features = report.get("data_stats", {}).get("features", [])
    if best_model in models:
        importance = models[best_model].get("feature_importance", {})
        method_labels = {
            "model": "XGBoost/model feature importance",
            "shap": "SHAP",
            "pearson": "Pearson",
            "spearman": "Spearman",
            "mutual_info": "Mutual Information",
        }
        available = [m for m in method_labels if m in importance]
        if not available:
            st.warning("Nenhum ranking de importancia disponivel neste relatorio.")
            return

        selected = st.selectbox(
            "Metodo de importancia",
            available,
            format_func=lambda method: method_labels.get(method, method),
        )
        scores = importance[selected]
        if len(scores) != len(features):
            st.warning("O relatorio nao possui o mesmo numero de features e scores.")
            return

        df_importance = pd.DataFrame({
            "Feature": features,
            "Importancia": scores,
        }).sort_values("Importancia", ascending=False)

        st.dataframe(df_importance, use_container_width=True, hide_index=True)
        st.bar_chart(df_importance.set_index("Feature"))


def render_raw_history_tab():
    st.subheader("Historico Bruto")
    df_training = get_training_dataframe()
    df_raw = get_optical_raw_dataframe()

    view = st.radio("Tabela", ["Treinamento IA", "Optica bruta"], horizontal=True)
    df = df_training if view == "Treinamento IA" else df_raw

    if df.empty:
        st.info("Nenhum registro encontrado.")
        return

    col1, col2 = st.columns(2)
    with col1:
        only_validated = st.checkbox("Somente validados", value=False)
    with col2:
        max_rows = st.number_input("Maximo de linhas", min_value=10, max_value=1000, value=100, step=10)

    if only_validated and "glicose_real" in df.columns:
        df = df[df["glicose_real"].notna()]

    df = df.head(int(max_rows))
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "Baixar CSV",
        data=csv,
        file_name=f"{view.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_ai_training_section():
    """Renderiza a area de IA dentro do dashboard principal."""
    st.header("Treinamento de IA - Hardware de Glicose")
    st.write("Acompanhe leituras do ESP32/MAX30102/BPW34, valide com glucosimetro e treine o modelo optico.")

    tabs = st.tabs([
        "Coleta do Hardware",
        "Validacao da Glicose",
        "Treinamento",
        "Analise Academica",
        "Historico Bruto",
    ])

    with tabs[0]:
        render_hardware_collection_tab()
    with tabs[1]:
        render_validation_tab()
    with tabs[2]:
        render_training_tab()
    with tabs[3]:
        render_analysis_tab()
    with tabs[4]:
        render_raw_history_tab()


def display_ai_training_page():
    """Compatibilidade com chamadas antigas."""
    render_ai_training_section()


if __name__ == "__main__":
    st.set_page_config(page_title="Treinamento IA", layout="wide")
    render_ai_training_section()
