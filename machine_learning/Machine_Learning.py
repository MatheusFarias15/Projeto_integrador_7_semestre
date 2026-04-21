# -*- coding: utf-8 -*-
"""
============================================================
  GLUCOSE ML PIPELINE — Pandas + Scikit-learn + XGBoost
  Comparação de Modelos | Seleção Automática pelo Melhor
============================================================

Módulos:
  1. GlucoseDataLoader          → Ingestão e validação
  2. FeatureEngineer            → Pré-processamento e correlação
  3. RegressionComparator       → GBT vs XGBoost vs RandomForest (Regressão)
  4. ClassificationComparator   → RF vs XGBoost vs Logistic (Classificação)
  5. ModelVisualizer            → Visualizações comparativas + melhor modelo
  6. main()                     → Orquestrador

Requisitos:
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTAÇÕES
# ─────────────────────────────────────────────────────────────
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("glucose_pipeline.log", encoding="utf-8"),
    ],
)

# ─────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
FIGURE_DPI   = 130

GLUCOSE_NORMAL_THRESHOLD   = 100.0
GLUCOSE_ELEVATED_THRESHOLD = 126.0
CLASS_LABELS               = ["alto", "elevado", "normal"]   # ordem alfabética do LabelEncoder

# Paleta clínica
PALETTE      = {"normal": "#4CAF50", "elevado": "#FF9800", "alto": "#F44336"}
DARK_BG      = "#1a1d27"
PANEL_BG     = "#12151e"
TEXT_COL     = "#e8e8e8"
GRID_COL     = "#2a2d3a"
MODEL_COLORS = {
    "GBT":                "#42A5F5",
    "XGBoost":            "#AB47BC",
    "RandomForest":       "#26A69A",
    "LogisticRegression": "#FF7043",
}


# ─────────────────────────────────────────────────────────────
#  1. INGESTÃO E VALIDAÇÃO
# ─────────────────────────────────────────────────────────────
class GlucoseDataLoader:
    """
    Carrega, sanitiza e valida o CSV.

    Parameters
    ----------
    filepath  : Caminho para o arquivo CSV.
    delimiter : Separador (padrão: ';').
    """

    def __init__(self, filepath: str | Path, delimiter: str = ";"):
        self.filepath  = Path(filepath)
        self.delimiter = delimiter
        self.log       = logging.getLogger(self.__class__.__name__)

    def load(self) -> pd.DataFrame:
        self.log.info("Carregando: %s", self.filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.filepath}")

        df = pd.read_csv(self.filepath, sep=self.delimiter)
        df = self._sanitize_columns(df)
        self._validate(df)
        self._report_missing(df)

        self.log.info("Shape: %s | Colunas: %s", df.shape, list(df.columns))
        return df

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        import re
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("DataFrame vazio após leitura.")
        if len(df) < 50:
            self.log.warning("Apenas %d linhas — resultados podem ser instáveis.", len(df))

    def _report_missing(self, df: pd.DataFrame) -> None:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            self.log.info("Sem valores ausentes.")
        else:
            self.log.warning("Valores ausentes:\n%s", missing.to_string())


# ─────────────────────────────────────────────────────────────
#  2. ENGENHARIA DE FEATURES
# ─────────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Pré-processa o DataFrame:
      - Codifica categóricas (LabelEncoder para binárias, get_dummies para multi)
      - Remove colunas de drop
      - Checa multicolinearidade
      - Gera heatmap de correlação

    Parameters
    ----------
    corr_threshold : Limiar de Pearson para alerta de multicolinearidade.
    drop_cols      : Colunas a remover antes do treino.
    cat_cols       : Colunas a codificar manualmente via cat.codes.
    """

    def __init__(
        self,
        corr_threshold: float = 0.85,
        drop_cols: list[str] | None = None,
        cat_cols: list[str] | None = None,
    ):
        self.corr_threshold = corr_threshold
        self.drop_cols      = drop_cols or []
        self.cat_cols       = cat_cols  or []
        self.log            = logging.getLogger(self.__class__.__name__)
        self._label_encoders: dict[str, LabelEncoder] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepara X (features) e y (target).

        Retorna
        -------
        X : pd.DataFrame  — features prontas para treino
        y : pd.Series     — target numérico (glicose)
        """
        df = df.copy().dropna()

        # Codifica colunas categóricas manuais via cat.codes
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes

        # Colunas object restantes → LabelEncoder (binário) ou get_dummies
        obj_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                    if c != target_col and c not in self.drop_cols]
        binary   = [c for c in obj_cols if df[c].nunique() <= 2]
        multi    = [c for c in obj_cols if df[c].nunique() > 2]

        for col in binary:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le

        if multi:
            df = pd.get_dummies(df, columns=multi, drop_first=True)

        # Remove colunas de drop
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        self.log.info("Features finais (%d): %s", len(X.columns), list(X.columns))
        self._check_multicollinearity(X)

        return X, y

    def _check_multicollinearity(self, X: pd.DataFrame) -> None:
        corr = X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high = [
            (c, r, upper.loc[r, c])
            for c in upper.columns for r in upper.index
            if pd.notna(upper.loc[r, c]) and upper.loc[r, c] >= self.corr_threshold
        ]
        if high:
            self.log.warning("⚠ Multicolinearidade (|r| ≥ %.2f):", self.corr_threshold)
            for a, b, v in sorted(high, key=lambda x: -x[2]):
                self.log.warning("   '%s' ↔ '%s' → r = %.4f", a, b, v)
        else:
            self.log.info("Sem multicolinearidade acima de %.2f.", self.corr_threshold)

    def plot_correlation_heatmap(self, X: pd.DataFrame, y: pd.Series, figsize=(13, 9)) -> None:
        data = X.copy()
        data["glicose"] = y.values
        corr = data.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8},
        )
        ax.set_title("Correlação de Pearson — Features + Glicose",
                     color=TEXT_COL, fontsize=13, pad=12)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        plt.tight_layout()
        plt.savefig("heatmap_correlacao.png", dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor=DARK_BG)
        plt.show()
        self.log.info("Heatmap salvo.")


# ─────────────────────────────────────────────────────────────
#  3. COMPARADOR DE REGRESSÃO
# ─────────────────────────────────────────────────────────────
class RegressionComparator:
    """
    Treina e compara 3 modelos de regressão:
      • GradientBoostingRegressor  (GBT)
      • XGBRegressor               (XGBoost)
      • RandomForestRegressor      (Random Forest)

    Seleciona automaticamente o melhor por R².

    Parameters
    ----------
    n_iter : Iterações do RandomizedSearchCV por modelo.
    """

    MODELS = {
        "GBT": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE, verbosity=0,
            eval_metric="rmse", tree_method="hist",
        ),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
    }

    PARAM_GRIDS = {
        "GBT": {
            "regressor__n_estimators":  [100, 200, 400],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__max_depth":     [3, 4, 5],
            "regressor__subsample":     [0.8, 1.0],
        },
        "XGBoost": {
            "regressor__n_estimators":  [100, 200, 400],
            "regressor__learning_rate": [0.05, 0.1, 0.2],
            "regressor__max_depth":     [3, 4, 6],
            "regressor__subsample":     [0.8, 1.0],
            "regressor__colsample_bytree": [0.7, 1.0],
        },
        "RandomForest": {
            "regressor__n_estimators": [100, 200, 400],
            "regressor__max_depth":    [None, 5, 10, 20],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
    }

    def __init__(self, n_iter: int = 15):
        self.n_iter   = n_iter
        self.log      = logging.getLogger(self.__class__.__name__)
        self.results_  : dict[str, dict] = {}
        self.best_name_: str             = ""
        self.best_pipe_: Pipeline | None = None
        self.X_test_  : pd.DataFrame | None = None
        self.y_test_  : pd.Series    | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressionComparator":
        X_train, self.X_test_, y_train, self.y_test_ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        self.log.info("Split: %d treino | %d teste", len(X_train), len(self.X_test_))

        for name, estimator in self.MODELS.items():
            self.log.info("─── Treinando: %s ───", name)
            pipe = Pipeline([
                ("scaler",    StandardScaler()),
                ("regressor", estimator),
            ])
            search = RandomizedSearchCV(
                pipe,
                param_distributions=self.PARAM_GRIDS[name],
                n_iter=self.n_iter,
                cv=CV_FOLDS,
                scoring="r2",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_
            y_pred    = best_pipe.predict(self.X_test_)

            rmse = np.sqrt(mean_squared_error(self.y_test_, y_pred))
            mae  = mean_absolute_error(self.y_test_, y_pred)
            r2   = r2_score(self.y_test_, y_pred)
            cv_r2 = cross_val_score(best_pipe, X_train, y_train, cv=CV_FOLDS, scoring="r2")

            self.results_[name] = {
                "pipeline":    best_pipe,
                "best_params": search.best_params_,
                "y_pred":      y_pred,
                "RMSE":        round(rmse, 4),
                "MAE":         round(mae, 4),
                "R²":          round(r2, 4),
                "CV R² Mean":  round(cv_r2.mean(), 4),
                "CV R² Std":   round(cv_r2.std(), 4),
            }
            self.log.info(
                "%s → R²=%.4f | RMSE=%.4f | MAE=%.4f | CV R²=%.4f±%.4f",
                name, r2, rmse, mae, cv_r2.mean(), cv_r2.std(),
            )

        # Seleciona melhor por R²
        self.best_name_ = max(self.results_, key=lambda n: self.results_[n]["R²"])
        self.best_pipe_ = self.results_[self.best_name_]["pipeline"]
        self.log.info("🏆 MELHOR REGRESSÃO: %s (R²=%.4f)", self.best_name_,
                      self.results_[self.best_name_]["R²"])
        return self

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, r in self.results_.items():
            rows.append({
                "Modelo": name,
                "R²":     r["R²"],
                "RMSE":   r["RMSE"],
                "MAE":    r["MAE"],
                "CV R²":  r["CV R² Mean"],
                "CV Std": r["CV R² Std"],
                "Melhor": "✅" if name == self.best_name_ else "",
            })
        return pd.DataFrame(rows).sort_values("R²", ascending=False)


# ─────────────────────────────────────────────────────────────
#  4. COMPARADOR DE CLASSIFICAÇÃO
# ─────────────────────────────────────────────────────────────
class ClassificationComparator:
    """
    Treina e compara 3 modelos de classificação multiclasse:
      • RandomForestClassifier   (Random Forest)
      • XGBClassifier            (XGBoost)
      • LogisticRegression       (Regressão Logística)

    Seleciona automaticamente o melhor por F1-macro.

    Classes clínicas (ADA Guidelines):
        normal  → glicose < 100 mg/dL
        elevado → 100 ≤ glicose < 126 mg/dL
        alto    → glicose ≥ 126 mg/dL
    """

    @staticmethod
    def classify_glucose(value: float) -> str:
        if value < GLUCOSE_NORMAL_THRESHOLD:
            return "normal"
        elif value < GLUCOSE_ELEVATED_THRESHOLD:
            return "elevado"
        return "alto"

    MODELS = {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, verbosity=0,
            eval_metric="mlogloss", tree_method="hist",
        ),
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000,
            class_weight="balanced",
        ),
    }

    PARAM_GRIDS = {
        "RandomForest": {
            "classifier__n_estimators":      [100, 200, 400],
            "classifier__max_depth":         [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features":      ["sqrt", "log2"],
        },
        "XGBoost": {
            "classifier__n_estimators":  [100, 200, 400],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__max_depth":     [3, 4, 6],
            "classifier__subsample":     [0.8, 1.0],
        },
        "LogisticRegression": {
            "classifier__C":       [0.01, 0.1, 1, 10, 100],
            "classifier__solver":  ["lbfgs", "saga"],
            "classifier__penalty": ["l2"],
        },
    }

    def __init__(self, n_iter: int = 12):
        self.n_iter    = n_iter
        self.log       = logging.getLogger(self.__class__.__name__)
        self.results_  : dict[str, dict] = {}
        self.best_name_: str             = ""
        self.best_pipe_: Pipeline | None = None
        self.label_encoder_: LabelEncoder = LabelEncoder()
        self.X_test_   : pd.DataFrame | None = None
        self.y_test_   : pd.Series    | None = None
        self.y_test_enc_: np.ndarray  | None = None

    def fit(self, X: pd.DataFrame, y_glucose: pd.Series) -> "ClassificationComparator":
        # Cria classes clínicas a partir dos valores de glicose
        y_classes = y_glucose.apply(self.classify_glucose)
        y_encoded = self.label_encoder_.fit_transform(y_classes)

        self.log.info("Distribuição de classes:\n%s",
                      pd.Series(y_classes).value_counts().to_string())

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        X_train, self.X_test_, y_train, self.y_test_enc_ = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
        self.y_test_ = self.label_encoder_.inverse_transform(self.y_test_enc_)

        for name, estimator in self.MODELS.items():
            self.log.info("─── Treinando: %s ───", name)
            pipe = Pipeline([
                ("scaler",     StandardScaler()),
                ("classifier", estimator),
            ])
            search = RandomizedSearchCV(
                pipe,
                param_distributions=self.PARAM_GRIDS[name],
                n_iter=self.n_iter,
                cv=skf,
                scoring="f1_macro",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
            search.fit(X_train, y_train)

            best_pipe  = search.best_estimator_
            y_pred_enc = best_pipe.predict(self.X_test_)
            y_pred     = self.label_encoder_.inverse_transform(y_pred_enc)

            acc       = accuracy_score(self.y_test_enc_, y_pred_enc)
            f1_macro  = f1_score(self.y_test_enc_, y_pred_enc, average="macro")
            f1_weight = f1_score(self.y_test_enc_, y_pred_enc, average="weighted")
            cv_f1     = cross_val_score(best_pipe, X_train, y_train, cv=skf, scoring="f1_macro")

            # ROC-AUC
            try:
                y_proba = best_pipe.predict_proba(self.X_test_)
                lb      = LabelBinarizer().fit(self.y_test_enc_)
                roc_auc = roc_auc_score(lb.transform(self.y_test_enc_), y_proba,
                                        multi_class="ovr", average="macro")
            except Exception:
                roc_auc = float("nan")

            self.results_[name] = {
                "pipeline":    best_pipe,
                "best_params": search.best_params_,
                "y_pred":      y_pred,
                "y_pred_enc":  y_pred_enc,
                "Accuracy":    round(acc, 4),
                "F1 Macro":    round(f1_macro, 4),
                "F1 Weighted": round(f1_weight, 4),
                "ROC-AUC":     round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
                "CV F1 Mean":  round(cv_f1.mean(), 4),
                "CV F1 Std":   round(cv_f1.std(), 4),
            }
            self.log.info(
                "%s → Acc=%.4f | F1=%.4f | AUC=%s | CV F1=%.4f±%.4f",
                name, acc, f1_macro, self.results_[name]["ROC-AUC"],
                cv_f1.mean(), cv_f1.std(),
            )

        # Seleciona melhor por Accuracy
        self.best_name_ = max(self.results_, key=lambda n: self.results_[n]["Accuracy"])
        self.best_pipe_ = self.results_[self.best_name_]["pipeline"]
        self.log.info("🏆 MELHOR CLASSIFICAÇÃO: %s (Acc=%.4f)", self.best_name_,
                      self.results_[self.best_name_]["Accuracy"])
        return self

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, r in self.results_.items():
            rows.append({
                "Modelo":      name,
                "Accuracy":    r["Accuracy"],
                "F1 Macro":    r["F1 Macro"],
                "F1 Weighted": r["F1 Weighted"],
                "ROC-AUC":     r["ROC-AUC"],
                "CV F1":       r["CV F1 Mean"],
                "CV Std":      r["CV F1 Std"],
                "Melhor":      "✅" if name == self.best_name_ else "",
            })
        return pd.DataFrame(rows).sort_values("Accuracy", ascending=False)


# ─────────────────────────────────────────────────────────────
#  5. VISUALIZADOR
# ─────────────────────────────────────────────────────────────
class ModelVisualizer:
    """
    Gera todas as visualizações comparativas e do melhor modelo.

    Gráficos:
      1. Comparação de métricas — Regressão (bar chart)
      2. Comparação de métricas — Classificação (bar chart)
      3. Análise de Resíduos — Melhor Regressor
      4. Importância de Features — Melhor Regressor (Permutation)
      5. Matriz de Confusão — Melhor Classificador
      6. Curvas ROC — Melhor Classificador
      7. Importância de Features — Melhor Classificador
      8. Dashboard final — Resultado da medição de glicose
    """

    log = logging.getLogger("ModelVisualizer")

    @staticmethod
    def _style(ax, title: str = "") -> None:
        ax.set_facecolor(DARK_BG)
        if title:
            ax.set_title(title, color=TEXT_COL, fontsize=11, pad=8, fontweight="bold")
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.5, linestyle="--")

    @staticmethod
    def _save(filename: str) -> None:
        plt.tight_layout()
        plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor=plt.gcf().get_facecolor())
        plt.show()
        ModelVisualizer.log.info("Salvo: %s", filename)

    # ──────────────────────────────────────────────────────────
    #  COMPARAÇÃO DE MODELOS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_regression_comparison(comparator: "RegressionComparator") -> None:
        """Bar chart comparando R², RMSE e MAE entre os 3 regressores."""
        ModelVisualizer.log.info("Gerando: Comparação de Regressores…")
        results = comparator.results_
        names   = list(results.keys())
        colors  = [MODEL_COLORS[n] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        fig.suptitle("Comparação de Modelos — Regressão de Glicose",
                     color=TEXT_COL, fontsize=14, fontweight="bold", y=1.02)

        metrics_plot = [("R²", True), ("RMSE", False), ("MAE", False)]

        for ax, (metric, higher_better) in zip(axes, metrics_plot):
            vals = [results[n][metric] for n in names]
            best_idx = vals.index(max(vals) if higher_better else min(vals))

            bars = ax.bar(names, vals, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5, width=0.5)

            # Destaca o melhor
            bars[best_idx].set_edgecolor("white")
            bars[best_idx].set_linewidth(2.5)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f"{val:.4f}", ha="center", va="bottom",
                        color=TEXT_COL, fontsize=9, fontweight="bold")

            direction = "↑ melhor" if higher_better else "↓ melhor"
            ModelVisualizer._style(ax, f"{metric} ({direction})")
            ax.set_xticklabels(names, rotation=10, ha="right")

        ModelVisualizer._save("comparacao_regressao.png")

    @staticmethod
    def plot_classification_comparison(comparator: "ClassificationComparator") -> None:
        """Bar chart comparando Accuracy, F1 Macro e ROC-AUC entre os 3 classificadores."""
        ModelVisualizer.log.info("Gerando: Comparação de Classificadores…")
        results = comparator.results_
        names   = list(results.keys())
        colors  = [MODEL_COLORS[n] for n in names]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        fig.suptitle("Comparação de Modelos — Classificação de Glicose",
                     color=TEXT_COL, fontsize=14, fontweight="bold", y=1.02)

        metrics_plot = ["Accuracy", "F1 Macro", "ROC-AUC"]

        for ax, metric in zip(axes, metrics_plot):
            vals = []
            for n in names:
                v = results[n][metric]
                vals.append(float(v) if v != "N/A" else 0.0)

            best_idx = vals.index(max(vals))
            bars = ax.bar(names, vals, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5, width=0.5)
            bars[best_idx].set_linewidth(2.5)

            for bar, val, raw in zip(bars, vals, [results[n][metric] for n in names]):
                label = f"{raw:.4f}" if raw != "N/A" else "N/A"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        label, ha="center", va="bottom",
                        color=TEXT_COL, fontsize=9, fontweight="bold")

            ax.set_ylim(0, min(1.15, max(vals) * 1.15))
            ModelVisualizer._style(ax, f"{metric} (↑ melhor)")
            ax.set_xticklabels(names, rotation=10, ha="right")

        ModelVisualizer._save("comparacao_classificacao.png")

    # ──────────────────────────────────────────────────────────
    #  ANÁLISE DO MELHOR REGRESSOR
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_residuals(comparator: "RegressionComparator") -> None:
        """Resíduos do melhor regressor: Scatter + Histograma."""
        name   = comparator.best_name_
        y_pred = comparator.results_[name]["y_pred"]
        y_true = comparator.y_test_

        ModelVisualizer.log.info("Gerando: Resíduos (%s)…", name)
        residuos = y_true.values - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        fig.suptitle(f"Análise de Resíduos — {name} (Melhor Regressor)",
                     color=TEXT_COL, fontsize=13, fontweight="bold")

        # Scatter
        color = MODEL_COLORS[name]
        axes[0].scatter(y_pred, residuos, alpha=0.5, color=color, s=22, edgecolors="none")
        axes[0].axhline(0, color="#F44336", lw=1.5, linestyle="--")
        z = np.polyfit(y_pred, residuos, 1)
        xl = np.linspace(y_pred.min(), y_pred.max(), 200)
        axes[0].plot(xl, np.poly1d(z)(xl), color="#FF9800", lw=1.8, label="Tendência")
        axes[0].set_xlabel("Previsto (mg/dL)"); axes[0].set_ylabel("Resíduo (Real − Previsto)")
        axes[0].legend(labelcolor=TEXT_COL, facecolor=DARK_BG, fontsize=8)
        ModelVisualizer._style(axes[0], "Previsto × Resíduo")

        # Histograma
        mu, sigma = residuos.mean(), residuos.std()
        axes[1].hist(residuos, bins=30, color=color, edgecolor="white",
                     linewidth=0.4, density=True, alpha=0.8)
        xn = np.linspace(residuos.min(), residuos.max(), 200)
        axes[1].plot(xn, (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((xn-mu)/sigma)**2),
                     color="#9C27B0", lw=2, label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
        axes[1].axvline(0, color="#F44336", lw=1.5, linestyle="--")
        axes[1].set_xlabel("Resíduo (mg/dL)"); axes[1].set_ylabel("Densidade")
        axes[1].legend(labelcolor=TEXT_COL, facecolor=DARK_BG, fontsize=8)
        ModelVisualizer._style(axes[1], "Distribuição dos Resíduos")

        ModelVisualizer._save("residuos_melhor_regressor.png")

    @staticmethod
    def plot_feature_importance_regression(comparator: "RegressionComparator") -> None:
        """Permutation Importance do melhor regressor."""
        name  = comparator.best_name_
        pipe  = comparator.results_[name]["pipeline"]
        ModelVisualizer.log.info("Gerando: Feature Importance Regressão (%s)…", name)

        result  = permutation_importance(
            pipe, comparator.X_test_, comparator.y_test_,
            n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1, scoring="r2"
        )
        feat_names = comparator.X_test_.columns.tolist()
        idx = result.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(idx)))
        ax.barh(
            [feat_names[i] for i in idx],
            result.importances_mean[idx],
            xerr=result.importances_std[idx],
            color=colors, edgecolor="white", linewidth=0.4, align="center",
        )
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_xlabel("Redução Média no R²", color=TEXT_COL)
        ModelVisualizer._style(ax, f"Importância de Features — {name} (Permutation)")
        ModelVisualizer._save("importancia_features_regressao.png")

    # ──────────────────────────────────────────────────────────
    #  ANÁLISE DO MELHOR CLASSIFICADOR
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_confusion_matrix(comparator: "ClassificationComparator") -> None:
        """Matriz de Confusão estilizada do melhor classificador."""
        name   = comparator.best_name_
        y_pred = comparator.results_[name]["y_pred"]
        y_true = comparator.y_test_
        labels = sorted(set(y_true))

        ModelVisualizer.log.info("Gerando: Matriz de Confusão (%s)…", name)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot  = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"

        fig, ax = plt.subplots(figsize=(8, 6), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        ax.set_facecolor(DARK_BG)
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor=GRID_COL, ax=ax,
                    cbar_kws={"label": "Contagem"})
        ax.tick_params(colors=TEXT_COL)
        ax.set_xlabel("Classe Predita", color=TEXT_COL, fontsize=11)
        ax.set_ylabel("Classe Real",    color=TEXT_COL, fontsize=11)
        ax.set_title(f"Matriz de Confusão — {name} (Melhor Classificador)",
                     color=TEXT_COL, fontsize=12, pad=10, fontweight="bold")
        ModelVisualizer._save("matriz_confusao.png")

    @staticmethod
    def plot_roc_curves(comparator: "ClassificationComparator") -> None:
        """Curvas ROC One-vs-Rest do melhor classificador."""
        name  = comparator.best_name_
        pipe  = comparator.results_[name]["pipeline"]
        ModelVisualizer.log.info("Gerando: Curvas ROC (%s)…", name)

        try:
            y_proba  = pipe.predict_proba(comparator.X_test_)
            classes  = pipe.classes_
            lb       = LabelBinarizer().fit(comparator.y_test_enc_)
            y_bin    = lb.transform(comparator.y_test_enc_)
            roc_colors = [MODEL_COLORS.get(name, "#42A5F5"), "#4CAF50", "#F44336"]

            fig, ax = plt.subplots(figsize=(8, 6), dpi=FIGURE_DPI)
            fig.patch.set_facecolor(PANEL_BG)
            ax.set_facecolor(DARK_BG)

            class_names_map = {0: "alto", 1: "elevado", 2: "normal"}
            for i, color in enumerate(roc_colors[:len(classes)]):
                if i < y_bin.shape[1]:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    auc_val     = roc_auc_score(y_bin[:, i], y_proba[:, i])
                    label       = class_names_map.get(classes[i], str(classes[i]))
                    ax.plot(fpr, tpr, color=color, lw=2.2,
                            label=f"Classe '{label}' (AUC = {auc_val:.3f})")

            ax.plot([0, 1], [0, 1], color=GRID_COL, lw=1.2, linestyle="--",
                    label="Aleatório (AUC = 0.500)")
            ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.set_xlabel("Taxa de Falsos Positivos (FPR)", color=TEXT_COL)
            ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", color=TEXT_COL)
            ax.set_title(f"Curvas ROC — {name} (One-vs-Rest)",
                         color=TEXT_COL, fontsize=12, pad=10, fontweight="bold")
            ax.legend(loc="lower right", facecolor=DARK_BG,
                      labelcolor=TEXT_COL, fontsize=9)
            ax.tick_params(colors=TEXT_COL)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ModelVisualizer._save("curvas_roc.png")
        except Exception as e:
            ModelVisualizer.log.warning("Não foi possível plotar ROC: %s", e)

    @staticmethod
    def plot_feature_importance_classification(comparator: "ClassificationComparator") -> None:
        """Importância de Features do melhor classificador."""
        name = comparator.best_name_
        pipe = comparator.results_[name]["pipeline"]
        ModelVisualizer.log.info("Gerando: Feature Importance Classificação (%s)…", name)

        feat_names = comparator.X_test_.columns.tolist()

        try:
            # Permutation importance (funciona para todos os modelos)
            result = permutation_importance(
                pipe, comparator.X_test_, comparator.y_test_enc_,
                n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1,
                scoring="f1_macro",
            )
            idx = result.importances_mean.argsort()[::-1]
            top = min(15, len(idx))

            fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
            fig.patch.set_facecolor(PANEL_BG)
            colors = plt.cm.plasma(np.linspace(0.2, 0.9, top))
            ax.barh(
                [feat_names[i] for i in idx[:top]],
                result.importances_mean[idx[:top]],
                xerr=result.importances_std[idx[:top]],
                color=colors, edgecolor="white", linewidth=0.4, align="center",
            )
            ax.invert_yaxis()
            ax.axvline(0, color="grey", lw=0.8, linestyle="--")
            ax.set_xlabel("Redução Média no F1-macro", color=TEXT_COL)
            ModelVisualizer._style(ax, f"Importância de Features — {name} (Permutation)")
            ModelVisualizer._save("importancia_features_classificacao.png")
        except Exception as e:
            ModelVisualizer.log.warning("Não foi possível plotar feature importance: %s", e)

    # ──────────────────────────────────────────────────────────
    #  DASHBOARD FINAL
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_final_dashboard(
        reg_comp: "RegressionComparator",
        clf_comp: "ClassificationComparator",
    ) -> None:
        """
        Dashboard completo mostrando:
          - Tabela comparativa dos 3 modelos (regressão e classificação)
          - Previsão final de glicose (melhor regressor)
          - Classificação final (melhor classificador)
          - Distribuição real vs prevista
        """
        ModelVisualizer.log.info("Gerando: Dashboard Final…")

        fig = plt.figure(figsize=(20, 14), dpi=FIGURE_DPI)
        fig.patch.set_facecolor(PANEL_BG)
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

        # ── Painel 0: Tabela Regressão ───────────────────────────────────
        ax0 = fig.add_subplot(gs[0, :2])
        ax0.set_facecolor(DARK_BG)
        ax0.axis("off")

        reg_data = []
        for name, r in reg_comp.results_.items():
            reg_data.append([
                f"{'🏆 ' if name == reg_comp.best_name_ else ''}{name}",
                f"{r['R²']:.4f}", f"{r['RMSE']:.4f}", f"{r['MAE']:.4f}",
                f"{r['CV R² Mean']:.4f} ± {r['CV R² Std']:.4f}",
            ])
        table = ax0.table(
            cellText=reg_data,
            colLabels=["Modelo", "R²", "RMSE", "MAE", "CV R² (mean ± std)"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False); table.set_fontsize(10)
        table.scale(1, 2.2)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(DARK_BG if row > 0 else "#1f2535")
            cell.set_text_props(color=TEXT_COL)
            cell.set_edgecolor(GRID_COL)
            if row > 0 and reg_data[row-1][0].startswith("🏆"):
                cell.set_facecolor("#1a3a2a")
        ax0.set_title("Regressão — Comparação de Modelos",
                      color=TEXT_COL, fontsize=12, pad=10, fontweight="bold")

        # ── Painel 1: Melhor Regressor — Scatter Real vs Previsto ────────
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.set_facecolor(DARK_BG)
        best_r     = reg_comp.best_name_
        y_pred_reg = reg_comp.results_[best_r]["y_pred"]
        y_true_reg = reg_comp.y_test_

        scatter_color = MODEL_COLORS[best_r]
        ax1.scatter(y_true_reg, y_pred_reg, alpha=0.5, s=20,
                    color=scatter_color, edgecolors="none")
        lims = [min(y_true_reg.min(), y_pred_reg.min()) - 2,
                max(y_true_reg.max(), y_pred_reg.max()) + 2]
        ax1.plot(lims, lims, "#F44336", lw=1.5, linestyle="--", label="Perfeito")
        ax1.set_xlabel("Real (mg/dL)"); ax1.set_ylabel("Previsto (mg/dL)")
        ax1.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        ModelVisualizer._style(ax1, f"Real vs Previsto — {best_r}")
        r2_val = reg_comp.results_[best_r]["R²"]
        ax1.text(0.05, 0.92, f"R² = {r2_val:.4f}", transform=ax1.transAxes,
                 color=TEXT_COL, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=scatter_color))

        # ── Painel 2: Tabela Classificação ──────────────────────────────
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.set_facecolor(DARK_BG)
        ax2.axis("off")

        clf_data = []
        for name, r in clf_comp.results_.items():
            clf_data.append([
                f"{'🏆 ' if name == clf_comp.best_name_ else ''}{name}",
                f"{r['Accuracy']:.4f}", f"{r['F1 Macro']:.4f}",
                f"{r['ROC-AUC']}", f"{r['CV F1 Mean']:.4f} ± {r['CV F1 Std']:.4f}",
            ])
        table2 = ax2.table(
            cellText=clf_data,
            colLabels=["Modelo", "Accuracy", "F1 Macro", "ROC-AUC", "CV F1 (mean ± std)"],
            loc="center", cellLoc="center",
        )
        table2.auto_set_font_size(False); table2.set_fontsize(10)
        table2.scale(1, 2.2)
        for (row, col), cell in table2.get_celld().items():
            cell.set_facecolor(DARK_BG if row > 0 else "#1f2535")
            cell.set_text_props(color=TEXT_COL)
            cell.set_edgecolor(GRID_COL)
            if row > 0 and clf_data[row-1][0].startswith("🏆"):
                cell.set_facecolor("#1a3a2a")
        ax2.set_title("Classificação — Comparação de Modelos",
                      color=TEXT_COL, fontsize=12, pad=10, fontweight="bold")

        # ── Painel 3: Distribuição das classes preditas ──────────────────
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor(DARK_BG)
        best_c     = clf_comp.best_name_
        y_pred_clf = clf_comp.results_[best_c]["y_pred"]
        y_true_clf = clf_comp.y_test_

        classes_order = ["normal", "elevado", "alto"]
        true_counts = pd.Series(y_true_clf).value_counts().reindex(classes_order, fill_value=0)
        pred_counts = pd.Series(y_pred_clf).value_counts().reindex(classes_order, fill_value=0)
        x_pos = np.arange(3)
        ax3.bar(x_pos - 0.2, true_counts.values, 0.35,
                label="Real",   color=[PALETTE[c] for c in classes_order], alpha=0.9)
        ax3.bar(x_pos + 0.2, pred_counts.values, 0.35,
                label="Predito", color=[PALETTE[c] for c in classes_order], alpha=0.5,
                edgecolor="white", linewidth=1)
        ax3.set_xticks(x_pos); ax3.set_xticklabels(classes_order)
        ax3.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        ModelVisualizer._style(ax3, f"Classes Real vs Predito — {best_c}")

        # ── Painel 4-6: Métricas comparativas radar-style (barras empilhadas) ─
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_facecolor(DARK_BG)
        ax4.axis("off")

        # Texto de resultado final
        reg_metrics_str = (
            f"R² = {reg_comp.results_[best_r]['R²']:.4f}   "
            f"RMSE = {reg_comp.results_[best_r]['RMSE']:.4f} mg/dL   "
            f"MAE = {reg_comp.results_[best_r]['MAE']:.4f} mg/dL"
        )
        clf_metrics_str = (
            f"Accuracy = {clf_comp.results_[best_c]['Accuracy']:.4f}   "
            f"F1-Macro = {clf_comp.results_[best_c]['F1 Macro']:.4f}   "
            f"ROC-AUC = {clf_comp.results_[best_c]['ROC-AUC']}"
        )

        ax4.text(0.5, 0.85, "RESULTADO FINAL DA MEDIÇÃO DE GLICOSE",
                 transform=ax4.transAxes, ha="center", va="center",
                 color=TEXT_COL, fontsize=15, fontweight="bold")

        ax4.text(0.5, 0.62,
                 f"🏆 Melhor Regressor:      {best_r}",
                 transform=ax4.transAxes, ha="center", va="center",
                 color=MODEL_COLORS[best_r], fontsize=13, fontweight="bold")
        ax4.text(0.5, 0.47, reg_metrics_str,
                 transform=ax4.transAxes, ha="center", va="center",
                 color=TEXT_COL, fontsize=11)

        ax4.text(0.5, 0.28,
                 f"🏆 Melhor Classificador:  {best_c}",
                 transform=ax4.transAxes, ha="center", va="center",
                 color=MODEL_COLORS[best_c], fontsize=13, fontweight="bold")
        ax4.text(0.5, 0.13, clf_metrics_str,
                 transform=ax4.transAxes, ha="center", va="center",
                 color=TEXT_COL, fontsize=11)

        # Linha decorativa
        for y_line in [0.74, 0.36]:
            ax4.axhline(y_line, color=GRID_COL, lw=1, linestyle="--",
                        transform=ax4.transAxes, xmin=0.05, xmax=0.95)

        fig.suptitle(
            "Glucose ML Pipeline — Comparação de Modelos & Resultado Final",
            color=TEXT_COL, fontsize=16, fontweight="bold", y=1.01,
        )
        ModelVisualizer._save("dashboard_final.png")


# ─────────────────────────────────────────────────────────────
#  6. ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main() -> None:
    log = logging.getLogger("main")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   GLUCOSE ML PIPELINE — Pandas + sklearn + XGBoost  ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    PATH_CSV = Path("Teste_ML_balanceado.csv")

    # ── 1. Carregamento ──────────────────────────────────────────────────────
    log.info("ETAPA 1/6 | Carregando dados…")
    loader = GlucoseDataLoader(PATH_CSV, delimiter=";")
    df     = loader.load()

    # ── 2. Engenharia de features ────────────────────────────────────────────
    log.info("ETAPA 2/6 | Engenharia de features…")
    fe = FeatureEngineer(
        corr_threshold=0.85,
        drop_cols=["data", "classe_glicose"],
        cat_cols=["humor", "treino"],
    )
    X, y = fe.fit_transform(df, target_col="glicose")
    fe.plot_correlation_heatmap(X, y)

    # ── 3. Regressão: treinamento e comparação ───────────────────────────────
    log.info("ETAPA 3/6 | Comparando modelos de REGRESSÃO…")
    log.info("  Modelos: GBT | XGBoost | RandomForest")
    reg_comp = RegressionComparator(n_iter=15)
    reg_comp.fit(X, y)

    log.info("\n%s", reg_comp.summary().to_string(index=False))

    # ── 4. Classificação: treinamento e comparação ───────────────────────────
    log.info("ETAPA 4/6 | Comparando modelos de CLASSIFICAÇÃO…")
    log.info("  Modelos: RandomForest | XGBoost | LogisticRegression")
    clf_comp = ClassificationComparator(n_iter=12)
    clf_comp.fit(X, y)

    log.info("\n%s", clf_comp.summary().to_string(index=False))

    # ── 5. Visualizações ─────────────────────────────────────────────────────
    log.info("ETAPA 5/6 | Gerando visualizações…")
    viz = ModelVisualizer()

    viz.plot_regression_comparison(reg_comp)
    viz.plot_classification_comparison(clf_comp)
    viz.plot_residuals(reg_comp)
    viz.plot_feature_importance_regression(reg_comp)
    viz.plot_confusion_matrix(clf_comp)
    viz.plot_roc_curves(clf_comp)
    viz.plot_feature_importance_classification(clf_comp)

    # ── 6. Dashboard final ───────────────────────────────────────────────────
    log.info("ETAPA 6/6 | Dashboard final…")
    viz.plot_final_dashboard(reg_comp, clf_comp)

    # ── SUMÁRIO ───────────────────────────────────────────────────────────────
    best_r = reg_comp.best_name_
    best_c = clf_comp.best_name_

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║                 SUMÁRIO FINAL                        ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  🏆 MELHOR REGRESSOR   → %-28s ║", best_r)
    for k in ["R²", "RMSE", "MAE"]:
        log.info("║     %-15s → %-28s ║", k, reg_comp.results_[best_r][k])
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  🏆 MELHOR CLASSIFICADOR → %-26s ║", best_c)
    for k in ["Accuracy", "F1 Macro", "ROC-AUC"]:
        log.info("║     %-15s → %-28s ║", k, clf_comp.results_[best_c][k])
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("Pipeline concluído. Logs em glucose_pipeline.log")


if __name__ == "__main__":
    main()

