"""
Script de Treinamento de IA para Estimativa de Glicose (versão CLI)
=====================================================================

Uso:
    python train_glucose_model_cli.py
    
Este script carrega dados da tabela ml_training_data e treina modelos de IA.
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importar o trainer customizado
sys.path.insert(0, str(Path(__file__).parent))
from machine_learning.glucose_trainer import GlucoseTrainer

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "health_database.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_training_data() -> pd.DataFrame:
    """Carrega dados de treinamento do banco de dados."""
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    
    # Importar modelo da rota
    from routes.glucose import MLTrainingData
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    # Carregar todos os registros
    records = db.query(MLTrainingData).all()
    
    if not records:
        print("ERRO: Nenhum dado de treinamento encontrado no banco de dados!")
        print("Execute o dashboard e use a aba 'Treinamento IA' para coletar dados.")
        return None
    
    # Converter para DataFrame
    data = []
    for record in records:
        data.append({
            'bpm': record.bpm,
            'spo2': record.spo2,
            'dc_ir': record.dc_ir,
            'ac_ir': record.ac_ir,
            'ir_max30102': record.ir_max30102,
            'red_max30102': record.red_max30102,
            'transmitancia_dc': record.transmitancia_dc,
            'transmitancia_ac': record.transmitancia_ac,
            'bpw34_raw': record.bpw34_raw,
            'bpw34_voltage': record.bpw34_voltage,
            'bpw34_current': record.bpw34_current,
            'bpw34_ac': record.bpw34_ac,
            'bpw34_dc': record.bpw34_dc,
            'bpw34_rms': record.bpw34_rms,
            'bpw34_peak': record.bpw34_peak,
            'bpw34_mean': record.bpw34_mean,
            'ir_940_intensity': record.ir_940_intensity,
            'ir_940_transmittance': record.ir_940_transmittance,
            'red_660': record.red_660,
            'temperatura': record.temperatura,
            'transmittance': record.transmittance,
            'absorbance': record.absorbance,
            'ratio_ir_trans': record.ratio_ir_trans,
            'pulsatile_index': record.pulsatile_index,
            'ir_ratio': record.ir_ratio,
            'ratio_ir_bpw34': record.ratio_ir_bpw34,
            'idade': record.idade,
            'peso': record.peso,
            'altura': record.altura,
            'imc': record.imc,
            'sexo': record.sexo,
            'ultima_refeicao_horas': record.ultima_refeicao_horas,
            'atividade_recente': record.atividade_recente,
            'glicose_real': record.glicose_real,
            'glicose_estimada': record.glicose_estimada,
            'erro_absoluto': record.erro_absoluto,
            'erro_percentual': record.erro_percentual,
        })
    
    db.close()
    
    df = pd.DataFrame(data)
    print(f"[OK] Carregados {len(df)} registros de treinamento")
    print(f"     Registros com glicose real: {df['glicose_real'].notna().sum()}")
    
    return df


def main():
    """Função principal do treinamento."""
    print("=" * 70)
    print("TREINADOR DE MODELOS DE IA PARA ESTIMATIVA DE GLICOSE")
    print("=" * 70)
    print()
    
    # Carregar dados
    print("[1/5] Carregando dados de treinamento...")
    df = load_training_data()
    
    if df is None or len(df) == 0:
        sys.exit(1)
    
    print()
    print("[2/5] Inicializando trainer...")
    trainer = GlucoseTrainer(test_size=0.2, random_state=42)
    
    # Preparar dados
    print("[3/5] Preparando dados...")
    try:
        X, y = trainer.load_data(df)
        print(f"     Features: {len(X.columns)} colunas, {len(X)} linhas")
        print(f"     Target (glicose real): min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")
    except ValueError as e:
        print(f"ERRO: {e}")
        sys.exit(1)
    
    print()
    print("[4/5] Treinando modelos (XGBoost, Random Forest, Gradient Boosting)...")
    results = trainer.train_all_models()
    
    # Exibir resultados
    print()
    print("-" * 70)
    print("RESULTADOS DE TREINAMENTO")
    print("-" * 70)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}")
        print(f"  R² Score:  {metrics['r2_score']:.4f}")
        print(f"  RMSE:      {metrics['rmse']:.2f} mg/dL")
        print(f"  MAE:       {metrics['mae']:.2f} mg/dL")
        print(f"  MAPE:      {metrics['mape']:.2f}%")
    
    # Selecionar melhor modelo
    best_model_name, best_model = trainer.select_best_model()
    best_metrics = trainer.metrics[best_model_name]
    
    print()
    print(f"[*] Melhor modelo: {best_model_name.upper()}")
    print(f"    R² = {best_metrics['r2_score']:.4f}")
    
    # Análise de features
    print()
    print("-" * 70)
    print(f"IMPORTÂNCIA DE FEATURES - {best_model_name.upper()}")
    print("-" * 70)
    
    feature_importance = trainer.feature_importance[best_model_name]
    features = trainer.X_train.columns.tolist()
    
    if 'model' in feature_importance:
        importance_scores = feature_importance['model']
        sorted_features = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
        print("\nModel Feature Importance (Top 10):")
        for feat, score in sorted_features[:10]:
            print(f"  {feat:25s} {score:.4f}")
    
    if 'shap' in feature_importance:
        importance_scores = feature_importance['shap']
        sorted_features = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
        print("\nSHAP Feature Importance (Top 10):")
        for feat, score in sorted_features[:10]:
            print(f"  {feat:25s} {score:.4f}")
    
    # Salvar modelos
    print()
    print("[5/5] Salvando modelos e relatório...")
    
    model_path = trainer.save_model(best_model_name)
    scaler_path = trainer.save_scaler()
    report_path = trainer.save_report()

    latest_model_path = MODELS_DIR / "hardware_glucose_model_latest.pkl"
    latest_scaler_path = MODELS_DIR / "hardware_glucose_scaler_latest.pkl"
    latest_features_path = MODELS_DIR / "hardware_feature_names_latest.json"
    latest_report_path = MODELS_DIR / "hardware_training_report_latest.json"

    shutil.copyfile(model_path, latest_model_path)
    shutil.copyfile(scaler_path, latest_scaler_path)
    shutil.copyfile(report_path, latest_report_path)
    with open(latest_features_path, "w", encoding="utf-8") as f:
        json.dump(trainer.X_train.columns.tolist(), f, indent=2, ensure_ascii=False)
    
    print(f"     Modelo: {model_path}")
    print(f"     Scaler: {scaler_path}")
    print(f"     Relatório: {report_path}")
    print(f"     Alias modelo hardware: {latest_model_path}")
    print(f"     Alias scaler hardware: {latest_scaler_path}")
    print(f"     Alias features hardware: {latest_features_path}")
    print(f"     Alias relatorio hardware: {latest_report_path}")
    
    print()
    print("=" * 70)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO")
    print("=" * 70)
    print()
    print("Próximos passos:")
    print("1. Use o dashboard Streamlit para coletar mais dados")
    print("2. Execute este script novamente para retreinar com mais dados")
    print("3. Monitore as métricas para avaliar a qualidade do modelo")
    print()


if __name__ == "__main__":
    main()
