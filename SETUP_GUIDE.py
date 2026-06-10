#!/usr/bin/env python3
"""
GUIA DE USO - Plataforma de Aprendizado Contínuo para Estimativa de Glicose
==============================================================================

Este script demonstra como usar o sistema completo passo a passo.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def print_section(title):
    """Imprime uma seção formatada."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_step(num, description):
    """Imprime um passo."""
    print(f"\n[PASSO {num}] {description}")
    print("-" * 80)

def main():
    BASE_DIR = Path(__file__).parent
    
    print_section("PLATAFORMA DE APRENDIZADO CONTÍNUO PARA ESTIMATIVA DE GLICOSE")
    print("""
    Uma solução completa para:
    ✓ Capturar dados de sensores (ESP32)
    ✓ Validar com medidas reais de glicose
    ✓ Treinar modelos de IA (XGBoost, Random Forest, Gradient Boosting)
    ✓ Analisar importância de features
    ✓ Fazer previsões em tempo real
    """)
    
    # PASSO 1
    print_step(1, "Preparar o Ambiente")
    print("""
    1.1) Instalar dependências:
         pip install -r requirements.txt
    
    1.2) Verificar que o ESP32 está conectado via USB:
         - Hardware necessário:
           • ESP32 (firmware em src/main.cpp)
           • MAX30105 (sensor pulsátil + O2)
           • ADS1115 + BPW34 (transmitância)
           • LED Infravermelha 940nm
         
         - Porta detectada automaticamente (padrão: COM5)
    """)
    
    # PASSO 2
    print_step(2, "Iniciar Captura de Sensores")
    print("""
    2.1) Em um terminal, iniciar o leitor serial:
         python serial_reader.py
    
    2.2) O script irá:
         - Detectar automaticamente porta COM5
         - Ler 5 sinais do ESP32: BPM, DC_IR, AC_IR, Transmitancia_DC, Transmitancia_AC
         - Enviar para Flask backend: POST http://localhost:5000/sensor-reading
         - Armazenar na tabela ml_training_data do banco de dados
    
    2.3) Mensagens esperadas:
         "Abrindo porta serial: COM5 @ 115200"
         "Leitura de sensor enviada com sucesso: {'bpm': 75.2, 'dc_ir': 45000, ...}"
    """)
    
    # PASSO 3
    print_step(3, "Iniciar Backend Flask")
    print("""
    3.1) Em outro terminal, iniciar o servidor Flask:
         python app.py
    
    3.2) O servidor irá:
         - Escutar em http://localhost:5000
         - Receber dados de sensores em POST /sensor-reading
         - Armazenar na tabela ml_training_data
         - Servir endpoints: /training-data (GET/POST), /training-data/<id> (PUT)
    
    3.3) Mensagens esperadas:
         "WARNING in app.run_simple: This is a development server..."
         "Running on http://127.0.0.1:5000"
    """)
    
    # PASSO 4
    print_step(4, "Iniciar Dashboard Streamlit")
    print("""
    4.1) Em um terceiro terminal, iniciar o dashboard:
         streamlit run dashboard.py
    
    4.2) Abra http://localhost:8501 no navegador
    
    4.3) Na aba "Treinamento IA" você poderá:
         - Ver últimas leituras de sensores
         - Registrar glicose real medida (com glucosímetro)
         - Atualizar dados demográficos (idade, peso, altura, etc.)
         - Visualizar histórico de registros
    """)
    
    # PASSO 5
    print_step(5, "Coletar Dados de Treinamento")
    print("""
    5.1) Procedimento de coleta (repetir múltiplas vezes):
         
         a) Execute serial_reader.py para capturar dados do sensor
         b) Espere 5-10 segundos para registro completo (5 sinais)
         c) Meça glicose real com glucosímetro
         d) Abra dashboard e use aba "Treinamento IA"
         e) Registre a glicose real no formulário
         f) Preencha dados demográficos (opcional)
         g) Clique "Salvar Glicose Real"
    
    5.2) Dados são salvos em: health_database.db (tabela ml_training_data)
    
    5.3) Mínimo recomendado para treinamento:
         - 20 registros para teste inicial
         - 50+ registros para modelo básico
         - 100+ registros para modelo robusto
    """)
    
    # PASSO 6
    print_step(6, "Treinar Modelos de IA")
    print("""
    6.1) Opção A - Via Dashboard:
         - Abra http://localhost:8501
         - Vá para aba "Treinamento IA" → "Treinamento"
         - Clique botão "▶️ INICIAR TREINAMENTO"
    
    6.2) Opção B - Via Terminal (CLI):
         python train_glucose_model_cli.py
    
    6.3) O treinamento irá:
         - Carregar dados de ml_training_data (apenas registros com glicose real)
         - Dividir em 80% treino, 20% teste
         - Treinar 3 modelos:
           • XGBoost (Gradient Boosting otimizado)
           • Random Forest (ensemble robusto)
           • Gradient Boosting (baseline poderoso)
         - Calcular métricas: R², RMSE, MAE, MAPE
         - Analisar importância de features (5 métodos):
           • Pearson correlation
           • Spearman correlation
           • Mutual Information
           • Model Feature Importance
           • SHAP values
         - Salvar melhor modelo em models/
    
    6.4) Saída esperada:
         =========================================================================
         TREINADOR DE MODELOS DE IA PARA ESTIMATIVA DE GLICOSE
         =========================================================================
         
         [1/5] Carregando dados de treinamento...
         [OK] Carregados 50 registros
              Registros com glicose real: 50
         
         [2/5] Inicializando trainer...
         [3/5] Preparando dados...
              Features: 14 colunas, 50 linhas
              Target (glicose real): min=80.0, max=180.0, mean=120.5
         
         [4/5] Treinando modelos...
         
         =========================================================================
         RESULTADOS DE TREINAMENTO
         =========================================================================
         
         XGBOOST
           R² Score:  0.8234
           RMSE:      12.45 mg/dL
           MAE:       9.23 mg/dL
           MAPE:      8.51%
         
         RANDOM_FOREST
           R² Score:  0.7891
           RMSE:      14.12 mg/dL
           MAE:       10.34 mg/dL
           MAPE:      9.78%
         
         GRADIENT_BOOSTING
           R² Score:  0.8156
           RMSE:      13.01 mg/dL
           MAE:       9.67 mg/dL
           MAPE:      8.92%
         
         [*] Melhor modelo: XGBOOST
             R² = 0.8234
         
         =========================================================================
         IMPORTÂNCIA DE FEATURES - XGBOOST
         =========================================================================
         
         Model Feature Importance (Top 10):
           ac_ir                0.1845
           transmitancia_ac     0.1621
           pulsatile_index      0.1234
           peso                 0.0987
           imc                  0.0876
           ... (resto omitido)
    """)
    
    # PASSO 7
    print_step(7, "Visualizar Resultados")
    print("""
    7.1) Abra http://localhost:8501 (dashboard)
    
    7.2) Vá para aba "Treinamento IA" → "Análise"
    
    7.3) Você verá:
         - Comparação de modelos (R², RMSE, MAE, MAPE)
         - Gráficos de desempenho
         - Importância de features (5 métodos diferentes)
         - Métricas do melhor modelo
    
    7.4) Relatório completo salvo em:
         models/glucose_training_report_YYYYMMDD_HHMMSS.json
    """)
    
    # PASSO 8
    print_step(8, "Retrainamento Contínuo")
    print("""
    8.1) Protocolo recomendado:
         
         a) Colete 10-20 novos registros com glicose real
         b) Execute train_glucose_model_cli.py novamente
         c) Compare R² com treinamento anterior
         d) Se R² melhorou: mantenha o novo modelo
         e) Se R² piorou: revise qualidade dos dados
    
    8.2) O sistema salva automaticamente:
         - Melhor modelo: models/glucose_model_xgboost_YYYYMMDD_HHMMSS.pkl
         - Scaler: models/glucose_scaler_YYYYMMDD_HHMMSS.pkl
         - Relatório: models/glucose_training_report_YYYYMMDD_HHMMSS.json
    """)
    
    # TABELA DE MAPEAMENTO
    print_section("ESTRUTURA DO BANCO DE DADOS")
    print("""
    Tabela: ml_training_data
    
    SINAIS DO SENSOR (capturados automaticamente):
    ├─ bpm                    (float) - Batidas por minuto
    ├─ dc_ir                  (float) - Componente DC do IR (MAX30105)
    ├─ ac_ir                  (float) - Componente AC do IR (MAX30105)
    ├─ transmitancia_dc       (float) - DC da transmitância (BPW34+ADS1115)
    └─ transmitancia_ac       (float) - AC da transmitância (BPW34+ADS1115)
    
    FEATURES CALCULADAS (derivadas):
    ├─ ratio_ir_trans         (float) - transmitancia_ac / ac_ir
    ├─ pulsatile_index        (float) - transmitancia_ac / transmitancia_dc
    └─ ir_ratio               (float) - ac_ir / dc_ir
    
    DADOS DEMOGRÁFICOS (preenchidos manualmente):
    ├─ idade                  (int)   - Anos
    ├─ peso                   (float) - Kg
    ├─ altura                 (float) - Metros
    ├─ imc                    (float) - peso / altura²
    ├─ sexo                   (str)   - Masculino/Feminino/Outro
    ├─ ultima_refeicao_horas  (float) - Horas desde última refeição
    └─ atividade_recente      (int)   - 0=nenhuma, 1=leve, 2=intensa
    
    ALVO E MÉTRICAS:
    ├─ glicose_real           (float) - Medida com glucosímetro (target)
    ├─ glicose_estimada       (float) - Previsão do modelo
    ├─ erro_absoluto          (float) - |glicose_real - glicose_estimada|
    └─ erro_percentual        (float) - (erro_absoluto / glicose_real) * 100
    
    METADADOS:
    ├─ id                     (int)   - Identificador único
    └─ created_at             (timestamp) - Data/hora de captura
    """)
    
    # TROUBLESHOOTING
    print_section("TROUBLESHOOTING")
    print("""
    PROBLEMA: "Nenhuma porta serial encontrada"
    SOLUÇÃO:
    • Verifique conexão USB entre ESP32 e computador
    • Instale driver CH9102 (USB-SERIAL): https://www.wch.cn/downloads
    • Mude porta manualmente: SERIAL_PORT=COM3 python serial_reader.py
    
    PROBLEMA: "Erro ao conectar ao servidor Flask"
    SOLUÇÃO:
    • Verifique se app.py está rodando em outro terminal
    • Confirme que está em http://localhost:5000
    • Verifique firewall do Windows
    
    PROBLEMA: "Nenhum dado de treinamento encontrado"
    SOLUÇÃO:
    • Execute serial_reader.py para capturar dados
    • Use dashboard para registrar glicose real
    • Verifique health_database.db: sqlite3 health_database.db
    
    PROBLEMA: "Dados insuficientes para treinamento"
    SOLUÇÃO:
    • Colete pelo menos 10 registros com glicose real
    • Realize múltiplas coletas (diferentes horários, atividades, refeições)
    • Verifique qualidade dos sinais do sensor
    
    PROBLEMA: "R² Score muito baixo (<0.5)"
    SOLUÇÃO:
    • Colete mais dados (100+ registros)
    • Verifique calibração do sensor
    • Valide medidas de glicose com aparelho confiável
    • Revise qualidade dos sinais (BPM entre 60-100, DC/AC valores válidos)
    """)
    
    # PRÓXIMOS PASSOS
    print_section("PRÓXIMOS PASSOS")
    print("""
    1. CURTO PRAZO (1-2 semanas):
       □ Configure hardware e software
       □ Colete 20+ registros validados
       □ Treine primeiro modelo
       □ Ajuste hiperparâmetros
    
    2. MÉDIO PRAZO (1-3 meses):
       □ Colete 100+ registros com variação
       □ Retreat modelo para melhorar R²
       □ Implemente validação cruzada
       □ Crie alertas de anomalias
    
    3. LONGO PRAZO (6+ meses):
       □ Validação clínica com pacientes
       □ Integração com aplicativo mobile
       □ Análise de tendências (A1C, TIR, etc.)
       □ Sistema de feedback automático
    """)
    
    print_section("FIM DO GUIA")
    print("\nPara dúvidas, consulte a documentação ou execute:")
    print("  python train_glucose_model_cli.py --help")
    print()


if __name__ == "__main__":
    main()
