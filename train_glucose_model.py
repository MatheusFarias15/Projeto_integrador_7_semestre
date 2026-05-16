"""
Script para Treinar o Modelo de Previsão de Glicose
Executa uma vez e salva o modelo para uso no dashboard
"""

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TrainGlucoseModel")

# Caminhos
ML_DIR = Path(__file__).parent / "machine_learning"
DATA_FILE = ML_DIR / "Teste_ML_balanceado.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

REGRESSOR_PATH = MODEL_DIR / "glucose_regressor.pkl"
SCALER_PATH = MODEL_DIR / "glucose_scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
GLUCOSE_CLASSES_PATH = MODEL_DIR / "glucose_classes.pkl"

RANDOM_STATE = 42


def train_glucose_model():
    """Treina o modelo de regressão para prever glicose"""
    
    log.info("=" * 60)
    log.info("TREINAMENTO DO MODELO DE PREVISÃO DE GLICOSE")
    log.info("=" * 60)
    
    # Verificar se arquivo de dados existe
    if not DATA_FILE.exists():
        log.error(f"Arquivo não encontrado: {DATA_FILE}")
        log.info("Usando dados de exemplo para criar o modelo...")
        df = create_sample_training_data()
    else:
        log.info(f"Carregando dados: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, sep=";")
        # Sanitizar nomes de colunas
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )
    
    log.info(f"Shape do dataset: {df.shape}")
    log.info(f"Colunas: {list(df.columns)}")
    
    # Remover linhas com valores ausentes
    df = df.dropna()
    log.info(f"Após remover NaN: {len(df)} linhas")
    
    # Codificar variáveis categóricas
    if "humor" in df.columns:
        df["humor"] = pd.Categorical(df["humor"]).codes
    if "treino" in df.columns:
        df["treino"] = pd.Categorical(df["treino"]).codes
    
    # Preparar features e target
    drop_cols = ["data", "classe_glicose", "glicose_pred"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    y = df["glicose"].copy()
    X = df.drop(columns=drop_cols + ["glicose"])
    
    log.info(f"Features ({len(X.columns)}): {list(X.columns)}")
    log.info(f"Target (glicose): min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")
    
    # Salvar nomes das features
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    log.info(f"Nomes das features salvos: {FEATURE_NAMES_PATH}")
    
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    log.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Pipeline de regressão
    log.info("\nTreinando pipeline de regressão...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", XGBRegressor(
            random_state=RANDOM_STATE,
            verbosity=0,
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
        )),
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Avaliação
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    mae = np.abs(y_test - y_pred).mean()
    
    log.info(f"  R² Score: {r2:.4f}")
    log.info(f"  RMSE: {rmse:.4f}")
    log.info(f"  MAE: {mae:.4f}")
    
    # Salvar modelo e scaler
    joblib.dump(pipeline, REGRESSOR_PATH)
    log.info(f"Modelo salvo: {REGRESSOR_PATH}")
    
    # Salvar classes de glicose para referência
    glucose_classes = {
        "normal": "< 100 mg/dL",
        "elevado": "100-126 mg/dL",
        "alto": ">= 126 mg/dL",
    }
    joblib.dump(glucose_classes, GLUCOSE_CLASSES_PATH)
    
    log.info("\n" + "=" * 60)
    log.info("✅ MODELO TREINADO COM SUCESSO!")
    log.info("=" * 60)


def create_sample_training_data():
    """Cria dados de exemplo se o arquivo não existir"""
    log.warning("Criando dados de exemplo para treino...")
    
    np.random.seed(RANDOM_STATE)
    n_samples = 200
    
    data = {
        "passos": np.random.randint(2000, 15000, n_samples),
        "sono_horas": np.random.uniform(5, 10, n_samples),
        "humor": np.random.choice([0, 1, 2], n_samples),  # Bom, Neutro, Ruim
        "kcal": np.random.randint(1500, 3000, n_samples),
        "carboidrato": np.random.randint(150, 350, n_samples),
        "proteina": np.random.randint(50, 150, n_samples),
        "gordura": np.random.randint(40, 120, n_samples),
        "agua_ml": np.random.randint(1000, 4000, n_samples),
        "treino": np.random.choice([0, 1, 2], n_samples),  # Nenhum, Leve, Intenso
        "deficit_kcal": np.random.randint(-500, 500, n_samples),
        # Gerar glicose com correlação aos dados
        "glicose": np.random.uniform(70, 180, n_samples),
    }
    
    df = pd.DataFrame(data)
    # Adicionar correlação: mais kcal/carboidrato = mais glicose
    df["glicose"] += (df["kcal"] - df["kcal"].mean()) * 0.01
    df["glicose"] += (df["carboidrato"] - df["carboidrato"].mean()) * 0.1
    df.loc[df["glicose"] < 60, "glicose"] = 60
    df.loc[df["glicose"] > 250, "glicose"] = 250
    
    return df


if __name__ == "__main__":
    try:
        train_glucose_model()
    except Exception as e:
        log.error(f"Erro ao treinar modelo: {e}", exc_info=True)
        sys.exit(1)
