# Plataforma de Aprendizado Continuo para Estimativa de Glicose

## Status: IMPLEMENTACAO CONCLUIDA (Tarefas 1-5 de 9)

### O que foi implementado?

**5 Tarefas Principais Completadas:**

1. **Tabela ml_training_data** (routes/glucose.py)
   - 20 campos estruturados
   - Sensores: BPM, DC_IR, AC_IR, Transmitancia_DC, Transmitancia_AC
   - Features: ratio_ir_trans, pulsatile_index, ir_ratio
   - Demograficos: idade, peso, altura, IMC, sexo, atividade
   - Alvo: glicose_real, glicose_estimada, erro_absoluto, erro_percentual

2. **Endpoints Flask** (routes/glucose.py)
   - POST /sensor-reading - registra leitura bruta
   - POST /training-data - registra com glicose real
   - GET /training-data - lista registros
   - PUT /training-data/<id> - atualiza registro

3. **Atualizacao serial_reader.py**
   - Nova funcao build_sensor_payload() que mapeia ESP32 -> API
   - Nova funcao send_sensor_reading() que POST para /sensor-reading
   - Compatibilidade mantida com endpoint legado /glucose

4. **Modulo de Treinamento** (machine_learning/glucose_trainer.py - 350 linhas)
   - Classe GlucoseTrainer com metodos completos:
     - load_data() - preparacao e normalizacao
     - train_xgboost() - XGBoost Regressor
     - train_random_forest() - Random Forest Regressor
     - train_gradient_boosting() - Gradient Boosting Regressor
     - evaluate_model() - calcula R2, RMSE, MAE, MAPE
     - compute_feature_importance() - 5 metodos:
       * Pearson correlation
       * Spearman correlation
       * Mutual Information
       * Model Feature Importance
       * SHAP values
     - select_best_model() - seleciona por R2 maximo
     - save_model() / save_scaler() - persistencia
     - generate_report() / save_report() - relatorios JSON

5. **CLI e Dashboard** (train_glucose_model_cli.py + pages_training_ai.py)
   - Script CLI para treinamento standalone
   - Dashboard Streamlit 3 abas:
     1. Coleta de Dados - registra glicose real, dados demograficos
     2. Treinamento - botao para iniciar treinamento CLI
     3. Analise - graficos, metricas, feature importance

### Arquivos Criados/Modificados

**NOVOS:**
- machine_learning/glucose_trainer.py (350 linhas)
- train_glucose_model_cli.py (150 linhas)
- pages_training_ai.py (400 linhas)
- SETUP_GUIDE.py (300 linhas)
- IMPLEMENTATION_SUMMARY.py
- README_ML_PLATFORM.md

**MODIFICADOS:**
- routes/glucose.py (+200 linhas, MLTrainingData model + 4 endpoints)
- serial_reader.py (+50 linhas, novo endpoint)
- requirements.txt (+4 dependencias: xgboost, matplotlib, seaborn, shap)

### Como usar?

**Terminal 1: Captura de Sensores**
```
python serial_reader.py
```

**Terminal 2: Backend Flask**
```
python app.py
```

**Terminal 3: Dashboard Streamlit**
```
streamlit run dashboard.py
```
Abra: http://localhost:8501

**Terminal 4: Treinamento (quando tiver 10+ registros com glicose_real)**
```
python train_glucose_model_cli.py
```

### Fluxo de Dados

```
ESP32 (COM5)
    |
    v (serial 5 sinais)
serial_reader.py
    |
    v (POST /sensor-reading)
Flask app.py
    |
    v (INSERT)
health_database.db (ml_training_data)
    |
    v (Dashboard coleta glicose_real)
    |
    v (PUT /training-data/<id>)
health_database.db (ml_training_data atualizado)
    |
    v
train_glucose_model_cli.py
    |
    v
GlucoseTrainer (XGBoost/RF/GB)
    |
    v
models/ (pkl + JSON report)
    |
    v
Dashboard (visualizacao)
```

### Dependencias Adicionadas

- xgboost
- matplotlib
- seaborn
- shap

### Tarefas Restantes (6-9)

- Tarefa 6: Feature engineering avancado
- Tarefa 7: Continuous learning/retraining automatico
- Tarefa 8: Correlacao avancada e feature selection
- Tarefa 9: Scientific reporting automatizado

### Metricas

- Arquivos criados: 4
- Arquivos modificados: 2
- Linhas adicionadas: 1500+
- Endpoints Flask: 4 novos
- Modelos de IA: 3 (XGBoost, Random Forest, Gradient Boosting)
- Metodos Feature Importance: 5
- Campos no banco: 20
- Abas do Dashboard: 3
- Metricas de avaliacao: 4 (R2, RMSE, MAE, MAPE)

### Para mais detalhes

Execute: python SETUP_GUIDE.py
