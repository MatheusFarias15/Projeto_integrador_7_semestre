"""
Dashboard de Rotina e Saúde com Previsão de Glicose — Streamlit
Aplicação para registrar dados de saúde e prever glicose usando Machine Learning

Para executar: streamlit run dashboard.py
Para treinar o modelo: python train_glucose_model.py
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# ============================================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Dashboard de Saúde com Previsão de Glicose",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONFIGURAÇÃO DO BANCO DE DADOS
# ============================================================================

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "health_database.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
MODELS_DIR = BASE_DIR / "models"

Base = declarative_base()


@st.cache_resource
def get_engine():
    """Retorna o engine do banco de dados (cached)"""
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@st.cache_resource
def get_session_maker():
    """Retorna a fábrica de sessões (cached)"""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@st.cache_resource
def load_glucose_model():
    """Carrega o modelo de previsão de glicose"""
    model_path = MODELS_DIR / "glucose_regressor.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    
    if model_path.exists() and feature_names_path.exists():
        try:
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            return model, feature_names
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar modelo: {e}")
            return None, None
    else:
        st.warning("⚠️ Modelo não treinado. Execute: python train_glucose_model.py")
        return None, None


engine = get_engine()
SessionLocal = get_session_maker()
glucose_model, feature_names = load_glucose_model()


class HealthRecord(Base):
    """Modelo de dados para registros de saúde"""
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    data = Column(DateTime, default=datetime.now, nullable=False)
    perfil = Column(String, nullable=False)
    passos = Column(Integer, nullable=False)
    sono_horas = Column(Float, nullable=False)
    humor = Column(Integer, nullable=False)
    kcal = Column(Integer, nullable=False)
    carboidrato = Column(Integer, nullable=False)
    proteina = Column(Integer, nullable=False)
    gordura = Column(Integer, nullable=False)
    agua_ml = Column(Integer, nullable=False)
    treino = Column(Integer, nullable=False)
    deficit_kcal = Column(Integer, nullable=False)
    glicose_prevista = Column(Float, nullable=True)  # Novo: glicose prevista pelo ML


@st.cache_resource
def init_db():
    """Inicializa o banco de dados"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return True


init_db()


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def get_db_session():
    """Obtém uma nova sessão do banco de dados"""
    return SessionLocal()


def predict_glucose(record_data: dict) -> float | None:
    """
    Prevê a glicose usando o modelo ML
    
    Parameters
    ----------
    record_data : dict
        Dicionário com os dados do formulário
    
    Returns
    -------
    float or None
        Valor previsto de glicose ou None se o modelo não estiver disponível
    """
    if glucose_model is None or feature_names is None:
        return None
    
    try:
        # Criar DataFrame com os dados na ordem esperada pelo modelo
        data_for_prediction = {}
        for feature in feature_names:
            if feature in record_data:
                data_for_prediction[feature] = record_data[feature]
            else:
                data_for_prediction[feature] = 0  # Valor padrão
        
        # Criar DataFrame
        df_input = pd.DataFrame([data_for_prediction])
        
        # Fazer previsão
        prediction = glucose_model.predict(df_input)[0]
        
        # Garantir que está dentro de limites razoáveis
        return max(50, min(300, float(prediction)))
    except Exception as e:
        st.error(f"Erro ao prever glicose: {e}")
        return None


def classify_glucose(glucose_value: float) -> tuple[str, str]:
    """
    Classifica o valor de glicose de acordo com padrões clínicos
    
    Returns
    -------
    tuple
        (classificação, cor) ex: ("Normal", "green")
    """
    if glucose_value < 100:
        return "Normal", "🟢"
    elif glucose_value < 126:
        return "Elevado", "🟡"
    else:
        return "Alto", "🔴"


def save_health_record(record_data: dict, glicose_prevista: float | None) -> bool:
    """Salva um registro de saúde no banco de dados"""
    try:
        db = get_db_session()
        new_record = HealthRecord(
            **record_data,
            glicose_prevista=glicose_prevista
        )
        db.add(new_record)
        db.commit()
        db.close()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar no banco de dados: {e}")
        return False


def get_all_records() -> pd.DataFrame:
    """Obtém todos os registros do banco de dados"""
    try:
        db = get_db_session()
        records = db.query(HealthRecord).order_by(HealthRecord.data.desc()).all()
        db.close()

        if not records:
            return pd.DataFrame()

        data = [
            {
                "Data": record.data.strftime("%d/%m/%Y %H:%M"),
                "Perfil": record.perfil,
                "Passos": record.passos,
                "Sono (h)": record.sono_horas,
                "Humor": ["Bom", "Neutro", "Ruim"][record.humor],
                "Kcal": record.kcal,
                "Carboidrato (g)": record.carboidrato,
                "Proteína (g)": record.proteina,
                "Gordura (g)": record.gordura,
                "Água (ml)": record.agua_ml,
                "Treino": ["Nenhum", "Leve", "Intenso"][record.treino],
                "Deficit Kcal": record.deficit_kcal,
                "Glicose Prevista": f"{record.glicose_prevista:.1f} mg/dL" if record.glicose_prevista else "-",
            }
            for record in records
        ]
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro ao buscar registros: {e}")
        return pd.DataFrame()


def get_records_for_chart() -> pd.DataFrame:
    """Obtém registros para gráficos"""
    try:
        db = get_db_session()
        records = db.query(HealthRecord).order_by(HealthRecord.data.asc()).all()
        db.close()

        if not records:
            return pd.DataFrame()

        data = [
            {
                "Data": record.data.strftime("%d/%m"),
                "Passos": record.passos,
                "Kcal": record.kcal,
                "Sono": record.sono_horas,
                "Água (ml)": record.agua_ml,
                "Glicose Prevista": record.glicose_prevista if record.glicose_prevista else None,
            }
            for record in records
        ]
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro ao buscar registros para gráfico: {e}")
        return pd.DataFrame()


def get_suggested_values(perfil: str) -> dict:
    """Retorna valores sugeridos baseado no perfil"""
    suggestions = {
        "Sedentário": {"passos": 3000, "sono_horas": 7.0},
        "Normal": {"passos": 7500, "sono_horas": 7.5},
        "Muito Ativo": {"passos": 12000, "sono_horas": 8.0},
    }
    return suggestions.get(perfil, {"passos": 5000, "sono_horas": 7.0})


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    """Função principal da aplicação"""
    
    st.title("📊 Dashboard de Saúde com Previsão de Glicose")
    
    # Aviso se modelo não está disponível
    if glucose_model is None:
        st.error(
            "❌ **Modelo de ML não carregado!**\n\n"
            "Execute o comando abaixo para treinar o modelo:\n"
            "```bash\npython train_glucose_model.py\n```"
        )
        st.info(
            "💡 O modelo será treinado com os dados de exemplo e poderá fazer previsões de glicose."
        )
    
    st.markdown("---")

    # Abas principais
    tab1, tab2, tab3 = st.tabs(["📝 Formulário", "📈 Dashboard", "📋 Histórico"])

    # ========================================================================
    # ABA 1: FORMULÁRIO COM PREVISÃO DE GLICOSE
    # ========================================================================

    with tab1:
        st.header("Registre seu Dia")
        st.markdown("Preencha as informações abaixo para registrar sua saúde e obter previsão de glicose.")

        if "perfil_selecionado" not in st.session_state:
            st.session_state.perfil_selecionado = "Normal"
        if "valores_sugeridos" not in st.session_state:
            st.session_state.valores_sugeridos = get_suggested_values("Normal")

        def update_perfil_values():
            st.session_state.perfil_selecionado = st.session_state.perfil_input
            st.session_state.valores_sugeridos = get_suggested_values(
                st.session_state.perfil_selecionado
            )

        col1, col2 = st.columns(2)

        with col1:
            perfil = st.selectbox(
                "🏃 Qual é seu estilo de vida atual?",
                options=["Sedentário", "Normal", "Muito Ativo"],
                index=1,
                key="perfil_input",
                on_change=update_perfil_values,
                help="Selecione seu nível de atividade física",
            )

        sugestoes = st.session_state.valores_sugeridos
        with col2:
            st.info(
                f"💡 **Valores sugeridos:**\n\n"
                f"- Passos: ~{sugestoes['passos']:,}\n"
                f"- Sono: ~{sugestoes['sono_horas']} h"
            )

        st.markdown("### Formulário de Entrada")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🚶 Atividade")
            passos = st.number_input(
                "Passos",
                value=int(sugestoes["passos"]),
                min_value=0,
                max_value=50000,
                step=100,
            )

            sono_horas = st.number_input(
                "Horas de Sono",
                value=sugestoes["sono_horas"],
                min_value=0.0,
                max_value=12.0,
                step=0.5,
            )

        with col2:
            st.subheader("😊 Bem-estar")
            humor = st.selectbox(
                "Humor",
                options=["Bom", "Neutro", "Ruim"],
                index=0,
                help="Como você se sente hoje?",
            )
            humor_value = {"Bom": 0, "Neutro": 1, "Ruim": 2}[humor]

            treino = st.selectbox(
                "Treino",
                options=["Nenhum", "Leve", "Intenso"],
                index=0,
            )
            treino_value = {"Nenhum": 0, "Leve": 1, "Intenso": 2}[treino]

        with col3:
            st.subheader("💧 Hidratação")
            agua_ml = st.number_input(
                "Água (ml)",
                value=2000,
                min_value=0,
                max_value=10000,
                step=100,
            )

        st.markdown("### Nutrição")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            kcal = st.number_input("Kcal", value=2000, min_value=0, max_value=10000, step=100)

        with col2:
            carboidrato = st.number_input("Carboidrato (g)", value=250, min_value=0, max_value=500, step=10)

        with col3:
            proteina = st.number_input("Proteína (g)", value=100, min_value=0, max_value=400, step=10)

        with col4:
            gordura = st.number_input("Gordura (g)", value=65, min_value=0, max_value=300, step=5)

        with col5:
            deficit_kcal = st.number_input(
                "Deficit Kcal",
                value=0,
                min_value=-2000,
                max_value=2000,
                step=100,
            )

        # Botão para salvar
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("💾 Salvar Registro e Prever Glicose", use_container_width=True, type="primary"):
                record_data = {
                    "perfil": perfil,
                    "passos": passos,
                    "sono_horas": sono_horas,
                    "humor": humor_value,
                    "kcal": kcal,
                    "carboidrato": carboidrato,
                    "proteina": proteina,
                    "gordura": gordura,
                    "agua_ml": agua_ml,
                    "treino": treino_value,
                    "deficit_kcal": deficit_kcal,
                }

                # Fazer previsão
                glicose_prevista = predict_glucose(record_data)
                
                # Salvar no banco
                if save_health_record(record_data, glicose_prevista):
                    st.success("✅ Registro salvo com sucesso!")
                    
                    # Mostrar previsão
                    if glicose_prevista is not None:
                        classificacao, emoji = classify_glucose(glicose_prevista)
                        st.markdown("---")
                        col_pred1, col_pred2 = st.columns(2)
                        
                        with col_pred1:
                            st.metric(
                                "📊 Glicose Prevista",
                                f"{glicose_prevista:.1f} mg/dL",
                                delta=classificacao
                            )
                        
                        with col_pred2:
                            status_text = f"{emoji} **{classificacao}**"
                            if classificacao == "Normal":
                                st.success(status_text)
                            elif classificacao == "Elevado":
                                st.warning(status_text)
                            else:
                                st.error(status_text)
                    else:
                        st.info("⚠️ Modelo não disponível para previsão")
                    
                    st.balloons()
                else:
                    st.error("❌ Erro ao salvar o registro.")

    # ========================================================================
    # ABA 2: DASHBOARD COM GRÁFICOS
    # ========================================================================

    with tab2:
        st.header("📈 Análise e Evolução")

        df_chart = get_records_for_chart()

        if not df_chart.empty:
            # Gráfico de Passos
            st.subheader("Evolução de Passos")
            st.line_chart(
                df_chart.set_index("Data")[["Passos"]],
                use_container_width=True,
            )

            # Gráficos lado a lado
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Evolução de Calorias")
                st.line_chart(
                    df_chart.set_index("Data")[["Kcal"]],
                    use_container_width=True,
                )

            with col2:
                st.subheader("Evolução de Sono")
                st.line_chart(
                    df_chart.set_index("Data")[["Sono"]],
                    use_container_width=True,
                )

            # Gráfico de Hidratação
            st.subheader("Evolução de Hidratação")
            st.line_chart(
                df_chart.set_index("Data")[["Água (ml)"]],
                use_container_width=True,
            )

            # Gráfico de Glicose Prevista (se disponível)
            glicose_data = df_chart.dropna(subset=["Glicose Prevista"])
            if not glicose_data.empty:
                st.subheader("Evolução de Glicose Prevista (ML)")
                st.line_chart(
                    glicose_data.set_index("Data")[["Glicose Prevista"]],
                    use_container_width=True,
                )

            # Estatísticas
            st.markdown("---")
            st.subheader("📊 Estatísticas Gerais")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Média de Passos", f"{df_chart['Passos'].mean():,.0f}")

            with col2:
                st.metric("Média de Kcal", f"{df_chart['Kcal'].mean():,.0f}")

            with col3:
                st.metric("Média de Sono", f"{df_chart['Sono'].mean():.1f}h")

            with col4:
                st.metric("Média de Água", f"{df_chart['Água (ml)'].mean():,.0f} ml")
            
            # Glicose média (se disponível)
            if not glicose_data.empty:
                st.metric("Média de Glicose Prevista", f"{glicose_data['Glicose Prevista'].mean():.1f} mg/dL")

        else:
            st.info("📭 Nenhum registro encontrado. Comece preenchendo o formulário!")

    # ========================================================================
    # ABA 3: HISTÓRICO
    # ========================================================================

    with tab3:
        st.header("📋 Histórico de Registros")

        df = get_all_records()

        if not df.empty:
            st.markdown(f"**Total de registros:** {len(df)}")

            # Filtros
            col1, col2, col3 = st.columns(3)

            with col1:
                filtro_perfil = st.multiselect(
                    "Filtrar por Perfil",
                    options=df["Perfil"].unique(),
                    default=df["Perfil"].unique(),
                )

            with col2:
                filtro_humor = st.multiselect(
                    "Filtrar por Humor",
                    options=df["Humor"].unique(),
                    default=df["Humor"].unique(),
                )

            with col3:
                filtro_treino = st.multiselect(
                    "Filtrar por Treino",
                    options=df["Treino"].unique(),
                    default=df["Treino"].unique(),
                )

            # Aplicar filtros
            df_filtered = df[
                (df["Perfil"].isin(filtro_perfil))
                & (df["Humor"].isin(filtro_humor))
                & (df["Treino"].isin(filtro_treino))
            ]

            st.markdown("---")

            if not df_filtered.empty:
                st.dataframe(df_filtered, use_container_width=True, hide_index=True)

                # Download CSV
                csv = df_filtered.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Baixar CSV",
                    data=csv,
                    file_name=f"historico_saude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("⚠️ Nenhum registro corresponde aos filtros selecionados.")

        else:
            st.info("📭 Nenhum registro encontrado. Comece preenchendo o formulário!")

    # ========================================================================
    # RODAPÉ
    # ========================================================================

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("💪 Dashboard de Saúde com ML")

    with col2:
        status_ml = "✅ Modelo Carregado" if glucose_model is not None else "⚠️ Modelo não disponível"
        st.caption(status_ml)

    with col3:
        st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":
    main()
