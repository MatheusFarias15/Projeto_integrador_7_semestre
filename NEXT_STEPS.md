# PROXIMOS PASSOS - Tarefas 6-9

## Resumo do que foi implementado (Tarefas 1-5)

CONCLUIDO:
✓ Tabela ml_training_data com 20 campos
✓ Endpoints Flask para captura (/sensor-reading, /training-data)
✓ serial_reader.py atualizado para novo endpoint
✓ Modulo GlucoseTrainer com 3 modelos de IA
✓ 5 metodos de feature importance (Pearson, Spearman, MI, Model, SHAP)
✓ CLI e Dashboard Streamlit para coleta e analise
✓ Relatorios JSON com metricas e features

## Proximas Tarefas (6-9)

### Tarefa 6: Feature Engineering Avancado
**Objetivo:** Criar features mais sofisticadas para melhorar precisao do modelo

**O que implementar:**
1. Features polinomiais (grau 2-3)
   - bpm^2, dc_ir^2, ac_ir^2, etc
   - bpm * ac_ir, dc_ir * transmitancia_ac, etc

2. Features de lag (valor anterior)
   - bpm_lag1, dc_ir_lag1 (usa registros anteriores)
   - Requer reorganizacao do DataFrame por timestamp

3. Features estatisticas por janela
   - media movel (média dos ultimos N registros)
   - desvio padrao da janela
   - minimo/maximo da janela

4. Features de dominancia frequencial
   - razao dc/ac para cada sinal
   - normalizacao min-max de cada sensor

5. Interacoes engineered
   - ratio_ir_trans * pulsatile_index
   - imc * ultima_refeicao_horas

**Onde implementar:**
- Metodo add_engineered_features() em GlucoseTrainer
- Chamado durante load_data()

**Metricas esperadas:**
- R2 score deve aumentar em 5-15%

---

### Tarefa 7: Continuous Learning & Retraining Automatico
**Objetivo:** Retreinar modelo automaticamente quando novos dados chegam

**O que implementar:**
1. Monitor de performance
   - Compara R2 atual vs modelo anterior
   - Detecta degradacao (R2 cai mais de 5%)

2. Trigger automatico de retrainamento
   - Apos cada 10 novos registros validados
   - Quando tempo desde ultimo training > 7 dias
   - Quando R2 do modelo atual < 0.7

3. Versionamento de modelos
   - Salva historico de modelos treinados
   - models/glucose_model_xgboost_v001.pkl
   - models/glucose_model_xgboost_v002.pkl

4. Metricas de evolucao
   - Grafico de R2 ao longo do tempo
   - Quantidade de registros no training set
   - Performance de cada modelo versao

5. Dashboard com status de retrainamento
   - Data do ultimo treinamento
   - R2 atual vs baseline
   - Alerta se performance degrada

**Onde implementar:**
- Classe ContinuousLearner (novo arquivo)
- Scheduler (APScheduler) para verificacoes periodicas
- Dashboard aba "Performance" exibe status

**Como testar:**
- Adicione 10-20 novos registros
- Execute: python continuous_learning_daemon.py
- Verifique models/ para novos arquivos

---

### Tarefa 8: Correlacao Avancada & Feature Selection
**Objetivo:** Automaticamente seleciomarem features mais importantes

**O que implementar:**
1. Matriz de correlacao de Pearson
   - Heatmap interativo (Plotly)
   - Detecta multicolinearidade (r > 0.9)
   - Remove features redundantes

2. Analise de Spearman
   - Para detectar relacoes nao-lineares
   - Comparar com Pearson para identificar features nao-lineares

3. Mutual Information (MI) avancado
   - Calcular MI com alvo para cada feature
   - Eliminar features com MI < threshold

4. Feature Selection Methods
   - Recursive Feature Elimination (RFE)
   - SelectKBest (top K features)
   - L1-based selection (Lasso/Ridge)

5. Visualizacoes
   - Heatmap de correlacao com cluster
   - Scatter plots de features importantes vs target
   - Grafico de feature selection evolution

6. Relatorio de redundancia
   - Lista features redundantes com r > 0.95
   - Sugestoes de remocao

**Onde implementar:**
- Metodo analyze_feature_correlations() em GlucoseTrainer
- Nova pagina "Analise de Features" no Dashboard
- Relatorio JSON separado com matriz de correlacao

**Metricas esperadas:**
- Reduzir numero de features sem perder performance
- Facilitar interpretacao do modelo

---

### Tarefa 9: Scientific Reporting Automatizado
**Objetivo:** Gerar relatorios profissionais em PDF/HTML

**O que implementar:**
1. Estrutura do relatorio
   - Titulo, data, versao do modelo
   - Resumo executivo (R2, RMSE, acuracia clinica)
   - Metodologia (dados, modelos, features)
   - Resultados (tabelas, graficos, metricas)
   - Discussao (interpretacao, limitacoes)
   - Conclusoes (recomendacoes, proximos passos)
   - Apendices (graficos detalhados, codigo)

2. Secoes tecnicas
   a) Dataset Description
      - Quantidade de amostras
      - Distribuicao temporal
      - Features: min/max/media/desvio
      - Valores faltantes

   b) Model Comparison
      - Tabela com metricas dos 3 modelos
      - Graficos de R2, RMSE, MAE, MAPE
      - Melhor modelo destacado

   c) Feature Importance
      - Grafico de importancia top 10
      - 5 metodos lado a lado (Pearson, Spearman, MI, Model, SHAP)
      - Interpreticacao de features importantes

   d) Error Analysis
      - Distribuicao de erros absolutos
      - Grafico residuos vs preditos
      - Casos de pior desempenho

   e) Clinical Validation
      - Matriz de confusao (se discretizado em ranges)
      - Sensitivity/Specificity para ranges clinicos
      - Analise de acordancia com glucosimetro

3. Geracao de relatorio
   - Opcao 1: PDF (ReportLab ou WeasyPrint)
   - Opcao 2: HTML (Jinja2 templates)
   - Opcao 3: Markdown (markdown library)

4. Integracao ao Dashboard
   - Botao "Gerar Relatorio" na aba Analise
   - Salva em reports/ com timestamp
   - Preview em abas antes de gerar PDF

5. Automacao
   - Gera relatorio automaticamente apos cada treinamento
   - Envia por email (SMTP) para coordenador
   - Arquiva versoes antigas

**Arquivos para criar:**
- machine_learning/scientific_report.py (classe ReportGenerator)
- templates/report_template.html (Jinja2)
- Adiciona aba "Relatorios" no Dashboard

**Exemplo de saida:**
```
Relatorio: Modelo XGBoost para Estimativa de Glicose
======================================================
Data do Relatorio: 2024-01-15 14:30
Versao do Modelo: 0.2.3
Dataset: 75 amostras validadas

METRICAS GERAIS
R2 Score:  0.8234
RMSE:      12.45 mg/dL
MAE:       9.23 mg/dL
MAPE:      8.51%

VALIDACAO CLINICA
Erro medio: 9.23 mg/dL
Erro percentual medio: 8.51%
Registros fora de range aceitavel (> 15 mg/dL): 3/75 (4%)

FEATURES MAIS IMPORTANTES
1. AC_IR (importancia: 18.45%)
2. Transmitancia_AC (16.21%)
3. Pulsatile_Index (12.34%)
... (top 10)

PROXIMOS PASSOS
- Coletar 25 mais amostras para validacao externa
- Testar em pacientes com diabetes tipo 2
- Integrar com aplicativo mobile
```

---

## Cronograma Estimado

- Tarefa 6 (Feature Engineering): 2-3 horas
- Tarefa 7 (Continuous Learning): 3-4 horas
- Tarefa 8 (Feature Selection): 2-3 horas
- Tarefa 9 (Scientific Reporting): 3-4 horas

Total: 10-14 horas para completar projeto

## Prioridade

1. **Alta**: Tarefa 9 (Scientific Reporting) - necessario para publicacao
2. **Alta**: Tarefa 7 (Continuous Learning) - essencial para producao
3. **Media**: Tarefa 6 (Feature Engineering) - melhora accuracia
4. **Baixa**: Tarefa 8 (Feature Selection) - otimizacao

## Comeco

Para comecar Tarefa 6:

```python
# Adicionar em GlucoseTrainer.load_data()
def add_engineered_features(self, X):
    X_eng = X.copy()
    
    # Features polinomiais
    for col in ['bpm', 'dc_ir', 'ac_ir']:
        if col in X_eng.columns:
            X_eng[f'{col}_squared'] = X_eng[col] ** 2
    
    # Media movel (janela de 3)
    for col in ['bpm', 'ac_ir']:
        if col in X_eng.columns:
            X_eng[f'{col}_ma3'] = X_eng[col].rolling(window=3, min_periods=1).mean()
    
    # Interacoes
    if 'imc' in X_eng.columns and 'ultima_refeicao_horas' in X_eng.columns:
        X_eng['imc_x_refeicao'] = X_eng['imc'] * X_eng['ultima_refeicao_horas']
    
    return X_eng
```

Boa sorte!
