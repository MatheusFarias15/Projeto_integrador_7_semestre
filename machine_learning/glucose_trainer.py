"""
Módulo de Treinamento de Modelos de IA para Estimativa de Glicose
==================================================================

Implementa:
- Carregamento de dados de treinamento
- Engenharia de features
- Treinamento de múltiplos modelos (XGBoost, Random Forest, Gradient Boosting)
- Análise de importância de features (Pearson, Spearman, MI, SHAP)
- Validação e métricas (R², RMSE, MAE, MAPE)
- Salvar/carregar modelos treinados
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class GlucoseTrainer:
    """Treinador de modelos de IA para estimativa de glicose."""
    
    # Features usadas para treinamento
    FEATURE_COLUMNS = [
        'bpm',
        'spo2',
        'dc_ir',
        'ac_ir',
        'ir_max30102',
        'red_max30102',
        'transmitancia_dc',
        'transmitancia_ac',
        'bpw34_raw',
        'bpw34_voltage',
        'bpw34_current',
        'bpw34_ac',
        'bpw34_dc',
        'bpw34_rms',
        'bpw34_peak',
        'bpw34_mean',
        'ir_940_intensity',
        'ir_940_transmittance',
        'red_660',
        'temperatura',
        'transmittance',
        'absorbance',
        'ratio_ir_trans',
        'pulsatile_index',
        'ir_ratio',
        'ratio_ir_bpw34',
        'idade',
        'peso',
        'altura',
        'imc',
        'ultima_refeicao_horas',
        'atividade_recente',
    ]
    
    TARGET_COLUMN = 'glicose_real'
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Inicializa o trainer.
        
        Args:
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Dicionário para armazenar modelos treinados
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepara dados para treinamento.
        
        Args:
            df: DataFrame com dados de treinamento
        
        Returns:
            X, y: Features e target
        """
        # Filtrar colunas válidas e remover NaN
        valid_features = [
            col for col in self.FEATURE_COLUMNS
            if col in df.columns and df[col].notna().any()
        ]

        if not valid_features:
            raise ValueError("Nenhuma feature valida encontrada para treinamento")
        
        # Usar apenas registros com glicose real (supervisionado)
        df_clean = df[valid_features + [self.TARGET_COLUMN]].dropna(subset=[self.TARGET_COLUMN]).copy()
        
        if len(df_clean) < 10:
            raise ValueError(f"Dados insuficientes: apenas {len(df_clean)} registros válidos")
        
        X = df_clean[valid_features].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        y = df_clean[self.TARGET_COLUMN]
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=valid_features)
        
        # Split treino/teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        
        return X_scaled, y
    
    def train_xgboost(self) -> xgb.XGBRegressor:
        """Treina modelo XGBoost."""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        return model
    
    def train_random_forest(self) -> RandomForestRegressor:
        """Treina modelo Random Forest."""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(self) -> GradientBoostingRegressor:
        """Treina modelo Gradient Boosting."""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = model
        return model
    
    def evaluate_model(self, model_name: str, model) -> Dict[str, float]:
        """
        Avalia modelo com métricas.
        
        Returns:
            Dicionário com R², RMSE, MAE, MAPE
        """
        y_pred = model.predict(self.X_test)
        
        metrics = {
            'r2_score': r2_score(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'mape': mean_absolute_percentage_error(self.y_test, y_pred),
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def compute_feature_importance(self, model_name: str, model) -> Dict[str, np.ndarray]:
        """
        Calcula importância de features usando múltiplos métodos.
        
        Returns:
            Dicionário com arrays de importância para cada método
        """
        importance = {}
        feature_names = self.X_train.columns.tolist()

        def normalize(scores: np.ndarray) -> np.ndarray:
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            total = np.sum(scores)
            if total == 0:
                return scores
            return scores / total
        
        # 1. Correlação de Pearson
        pearson_corr = np.array([
            abs(pearsonr(self.X_train[feat], self.y_train)[0])
            for feat in feature_names
        ])
        importance['pearson'] = normalize(pearson_corr)
        
        # 2. Correlação de Spearman
        spearman_corr = np.array([
            abs(spearmanr(self.X_train[feat], self.y_train)[0])
            for feat in feature_names
        ])
        importance['spearman'] = normalize(spearman_corr)
        
        # 3. Informação Mútua
        mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=self.random_state)
        importance['mutual_info'] = normalize(mi_scores)
        
        # 4. Feature Importance do modelo
        if hasattr(model, 'feature_importances_'):
            importance['model'] = model.feature_importances_
        
        # 5. SHAP (para modelos tree-based)
        try:
            if model_name == 'xgboost':
                explainer = shap.TreeExplainer(model)
            elif model_name in ['random_forest', 'gradient_boosting']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(self.X_train, 100))
            
            shap_values = explainer.shap_values(self.X_test.iloc[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            importance['shap'] = normalize(np.abs(shap_values).mean(axis=0))
        except Exception as e:
            print(f"Aviso: SHAP falhou para {model_name}: {str(e)}")
        
        self.feature_importance[model_name] = importance
        return importance
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Treina todos os modelos e retorna resultados."""
        results = {}
        
        # Treinar XGBoost
        print("Treinando XGBoost...")
        model_xgb = self.train_xgboost()
        metrics_xgb = self.evaluate_model('xgboost', model_xgb)
        importance_xgb = self.compute_feature_importance('xgboost', model_xgb)
        results['xgboost'] = {
            'model': model_xgb,
            'metrics': metrics_xgb,
            'importance': importance_xgb
        }
        
        # Treinar Random Forest
        print("Treinando Random Forest...")
        model_rf = self.train_random_forest()
        metrics_rf = self.evaluate_model('random_forest', model_rf)
        importance_rf = self.compute_feature_importance('random_forest', model_rf)
        results['random_forest'] = {
            'model': model_rf,
            'metrics': metrics_rf,
            'importance': importance_rf
        }
        
        # Treinar Gradient Boosting
        print("Treinando Gradient Boosting...")
        model_gb = self.train_gradient_boosting()
        metrics_gb = self.evaluate_model('gradient_boosting', model_gb)
        importance_gb = self.compute_feature_importance('gradient_boosting', model_gb)
        results['gradient_boosting'] = {
            'model': model_gb,
            'metrics': metrics_gb,
            'importance': importance_gb
        }
        
        return results
    
    def select_best_model(self) -> Tuple[str, Any]:
        """Seleciona melhor modelo baseado em R² score."""
        best_model_name = max(
            self.metrics.keys(),
            key=lambda m: self.metrics[m]['r2_score']
        )
        return best_model_name, self.models[best_model_name]
    
    def save_model(self, model_name: str, filepath: Path = None) -> Path:
        """Salva modelo treinado."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f"glucose_model_{model_name}_{timestamp}.pkl"
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        return filepath
    
    def save_scaler(self, filepath: Path = None) -> Path:
        """Salva scaler para normalização de features."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f"glucose_scaler_{timestamp}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return filepath
    
    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório completo de treinamento."""
        best_model_name, best_model = self.select_best_model()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'total_records': len(self.X_train) + len(self.X_test),
                'train_records': len(self.X_train),
                'test_records': len(self.X_test),
                'features': self.X_train.columns.tolist(),
            },
            'models': {},
            'best_model': best_model_name,
            'best_metrics': self.metrics[best_model_name],
        }
        
        # Adicionar resultados de cada modelo
        for model_name in self.models.keys():
            report['models'][model_name] = {
                'metrics': self.metrics.get(model_name, {}),
                'feature_importance': {
                    method: arr.tolist() 
                    for method, arr in self.feature_importance.get(model_name, {}).items()
                }
            }
        
        return report
    
    def save_report(self, filepath: Path = None) -> Path:
        """Salva relatório em JSON."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODELS_DIR / f"glucose_training_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath


def load_model(filepath: Path) -> Any:
    """Carrega modelo salvo."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_scaler(filepath: Path) -> StandardScaler:
    """Carrega scaler salvo."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
