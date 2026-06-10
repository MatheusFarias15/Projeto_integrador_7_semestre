"""
Página de Treinamento de IA para o Dashboard Streamlit
========================================================

Esta página permite:
- Visualizar últimas leituras de sensores
- Registrar glicose real medida
- Treinar modelos de IA
- Visualizar importância de features
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "health_database.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"


def get_db_session():
    """Retorna uma sessão do banco de dados."""
    from routes.glucose import MLTrainingData
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal(), MLTrainingData


def display_ai_training_page():
    """Exibe página de treinamento de IA."""
    
    st.header("🤖 Treinamento de IA - Estimativa de Glicose")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Coleta de Dados", "Treinamento", "Análise"])
    
    # ====================================================================
    # TAB 1: COLETA DE DADOS
    # ====================================================================
    with tab1:
        st.subheader("Coletar Novo Registro de Dados")
        
        db, MLTrainingData = get_db_session()
        
        # Buscar último registro de sensor não validado
        last_record = db.query(MLTrainingData).filter(
            MLTrainingData.glicose_real == None
        ).order_by(MLTrainingData.created_at.desc()).first()
        
        if last_record:
            st.success("✓ Encontrado registro de sensor sem validação!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Sinais do Sensor**")
                st.metric("BPM", f"{last_record.bpm:.0f}" if last_record.bpm else "N/A")
                st.metric("DC IR", f"{last_record.dc_ir:.0f}" if last_record.dc_ir else "N/A")
                st.metric("AC IR", f"{last_record.ac_ir:.0f}" if last_record.ac_ir else "N/A")
                st.metric("Transmitância DC", f"{last_record.transmitancia_dc:.0f}" if last_record.transmitancia_dc else "N/A")
                st.metric("Transmitância AC", f"{last_record.transmitancia_ac:.0f}" if last_record.transmitancia_ac else "N/A")
            
            with col2:
                st.info("**Features Calculadas**")
                st.metric("Ratio IR/Trans", f"{last_record.ratio_ir_trans:.4f}" if last_record.ratio_ir_trans else "N/A")
                st.metric("Pulsatile Index", f"{last_record.pulsatile_index:.4f}" if last_record.pulsatile_index else "N/A")
                st.metric("IR Ratio", f"{last_record.ir_ratio:.4f}" if last_record.ir_ratio else "N/A")
            
            st.divider()
            st.subheader("Validar com Medida Real de Glicose")
            
            col1, col2 = st.columns(2)
            
            with col1:
                glicose_real = st.number_input(
                    "Glicose Real (mg/dL)",
                    min_value=30.0,
                    max_value=400.0,
                    value=100.0,
                    step=1.0,
                    help="Valor medido com glucosímetro"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("✓ Salvar Glicose Real", key="save_glucose"):
                    # Enviar PUT para atualizar registro
                    try:
                        response = requests.put(
                            f"http://localhost:5000/training-data/{last_record.id}",
                            json={"glicose_real": glicose_real},
                            timeout=5
                        )
                        response.raise_for_status()
                        st.success(f"✓ Registrado: glicose real = {glicose_real} mg/dL")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")
            
            # Formulário para dados demográficos e contextuais
            st.divider()
            st.subheader("Dados Demográficos e Contextuais (Opcional)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                idade = st.number_input("Idade (anos)", min_value=0, max_value=120, value=30)
                peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
            
            with col2:
                altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
                sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
            
            with col3:
                ultima_refeicao = st.number_input("Última refeição (horas atrás)", min_value=0.0, max_value=24.0, value=2.0)
                atividade = st.selectbox(
                    "Atividade recente",
                    ["Nenhuma (0)", "Leve (1)", "Intensa (2)"],
                    help="Exercício físico nas últimas 2 horas"
                )
            
            atividade_valor = int(atividade.split("(")[1].rstrip(")"))
            
            if st.button("📝 Atualizar Dados Demográficos", key="update_demographics"):
                try:
                    response = requests.put(
                        f"http://localhost:5000/training-data/{last_record.id}",
                        json={
                            "idade": idade,
                            "peso": peso,
                            "altura": altura,
                            "sexo": sexo,
                            "ultima_refeicao_horas": ultima_refeicao,
                            "atividade_recente": atividade_valor,
                        },
                        timeout=5
                    )
                    response.raise_for_status()
                    st.success("✓ Dados demográficos atualizados!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao atualizar: {e}")
        
        else:
            st.info("📊 Nenhum registro de sensor pendente. Execute serial_reader.py para capturar dados.")
        
        # Mostrar histórico
        st.divider()
        st.subheader("Histórico de Registros Coletados")
        
        records = db.query(MLTrainingData).order_by(MLTrainingData.created_at.desc()).limit(20).all()
        
        if records:
            data = []
            for r in records:
                data.append({
                    'Data': r.created_at.strftime("%Y-%m-%d %H:%M"),
                    'BPM': f"{r.bpm:.0f}" if r.bpm else "-",
                    'Glicose Real': f"{r.glicose_real:.0f}" if r.glicose_real else "Pendente",
                    'Validado': "✓" if r.glicose_real else "✗",
                })
            
            df_records = pd.DataFrame(data)
            st.dataframe(df_records, use_container_width=True)
        
        db.close()
    
    # ====================================================================
    # TAB 2: TREINAMENTO
    # ====================================================================
    with tab2:
        st.subheader("Treinar Modelos de IA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(
                """
                **Processo de Treinamento:**
                
                1. Carrega dados validados (com glicose real) do banco de dados
                2. Divide em 80% treino / 20% teste
                3. Treina 3 modelos: XGBoost, Random Forest, Gradient Boosting
                4. Calcula métricas: R², RMSE, MAE, MAPE
                5. Analisa importância de features (5 métodos)
                6. Salva o melhor modelo
                """
            )
        
        with col2:
            if st.button("▶️ INICIAR TREINAMENTO", key="train", use_container_width=True):
                st.session_state['training'] = True
        
        if st.session_state.get('training'):
            with st.spinner("⏳ Treinando modelos (isso pode levar alguns minutos)..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["python", "train_glucose_model_cli.py"],
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        st.success("✓ Treinamento concluído com sucesso!")
                        st.text(result.stdout)
                    else:
                        st.error("❌ Erro durante treinamento:")
                        st.text(result.stderr)
                
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Treinamento excedeu tempo limite")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")
                finally:
                    st.session_state['training'] = False
    
    # ====================================================================
    # TAB 3: ANÁLISE
    # ====================================================================
    with tab3:
        st.subheader("Análise de Features e Métricas")
        
        # Carregar relatório mais recente
        models_dir = BASE_DIR / "models"
        
        if models_dir.exists():
            reports = sorted(models_dir.glob("glucose_training_report_*.json"), reverse=True)
            
            if reports:
                latest_report_path = reports[0]
                
                with open(latest_report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                # Mostrar data do treinamento
                st.info(f"Relatório de: {report['timestamp']}")
                
                # Métricas dos modelos
                st.subheader("Comparação de Modelos")
                
                metrics_data = []
                for model_name, model_info in report['models'].items():
                    metrics = model_info['metrics']
                    metrics_data.append({
                        'Modelo': model_name,
                        'R²': metrics.get('r2_score', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'MAPE': metrics.get('mape', 0),
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**R² Score (mais alto é melhor)**")
                    fig_r2 = go.Figure(data=[
                        go.Bar(x=df_metrics['Modelo'], y=df_metrics['R²'], marker_color='#1f77b4')
                    ])
                    fig_r2.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    st.write("**RMSE (mais baixo é melhor)**")
                    fig_rmse = go.Figure(data=[
                        go.Bar(x=df_metrics['Modelo'], y=df_metrics['RMSE'], marker_color='#ff7f0e')
                    ])
                    fig_rmse.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Melhor modelo
                best_model = report['best_model']
                best_metrics = report['best_metrics']
                
                st.divider()
                st.subheader(f"Melhor Modelo: {best_model.upper()}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{best_metrics.get('r2_score', 0):.4f}")
                col2.metric("RMSE (mg/dL)", f"{best_metrics.get('rmse', 0):.2f}")
                col3.metric("MAE (mg/dL)", f"{best_metrics.get('mae', 0):.2f}")
                col4.metric("MAPE (%)", f"{best_metrics.get('mape', 0):.2f}")
                
                # Importância de features
                st.divider()
                st.subheader(f"Importância de Features - {best_model.upper()}")
                
                if best_model in report['models']:
                    feature_imp = report['models'][best_model]['feature_importance']
                    
                    # Selecionar qual método mostrar
                    methods = list(feature_imp.keys())
                    selected_method = st.selectbox("Método de Importância", methods)
                    
                    if selected_method in feature_imp:
                        importance_array = feature_imp[selected_method]
                        # Assumindo que features estão na mesma ordem
                        feature_names = [
                            'BPM', 'DC_IR', 'AC_IR', 'Transmitancia_DC', 'Transmitancia_AC',
                            'Ratio_IR_Trans', 'Pulsatile_Index', 'IR_Ratio',
                            'Idade', 'Peso', 'Altura', 'IMC',
                            'Ultima_Refeicao_Horas', 'Atividade_Recente'
                        ]
                        
                        if len(importance_array) == len(feature_names):
                            df_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Importância': importance_array
                            }).sort_values('Importância', ascending=True)
                            
                            fig_feat = go.Figure(data=[
                                go.Bar(
                                    x=df_importance['Importância'],
                                    y=df_importance['Feature'],
                                    orientation='h',
                                    marker_color='#2ca02c'
                                )
                            ])
                            fig_feat.update_layout(height=500, showlegend=False)
                            st.plotly_chart(fig_feat, use_container_width=True)
            
            else:
                st.warning("📊 Nenhum relatório de treinamento encontrado. Execute o treinamento primeiro.")
        
        else:
            st.warning("📁 Diretório de modelos não existe ainda.")


if __name__ == "__main__":
    st.set_page_config(page_title="Treinamento IA", layout="wide")
    display_ai_training_page()
