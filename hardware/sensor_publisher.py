import time
import requests
import json

# URL da sua API Flask (se estiver rodando localmente na mesma rede, use o IP do PC, ex: http://192.166.X.X:5000/glucose)
# Se já estiver na nuvem (AWS/GCP), use a URL pública.
API_URL = "http://127.0.0.1:5000/glucose"

def read_sensor_data():
    """
    Simula a leitura do sensor AS7265x via I2C e o processamento inicial.
    Na prática, você importará a biblioteca do sensor (ex: qwiic_as7265x) aqui.
    """
    # Exemplo de captura de algumas das 18 bandas do sensor espectral
    espectro_capturado = {
        "410nm": 0.12, 
        "435nm": 0.15, 
        "560nm": 0.35, 
        "940nm": 0.85 
    }
    
    # Aqui entraria o seu modelo de Machine Learning (joblib) para calcular a glicose
    # glicose_calculada = modelo.predict(espectro_capturado)
    glicose_calculada = 105.5 # Valor simulado pós-processamento
    
    temperatura_caixa = 25.3 # Leitura do sensor de temperatura interno
    
    return {
        "glucose_mg_dl": glicose_calculada,
        "spectral_transmittance_data": json.dumps(espectro_capturado), # Converte o dict para string JSON
        "box_temperature_celsius": temperatura_caixa,
        "measurement_phase": "in_vitro", # ou "in_vivo"
        "real_concentration": 105.0 # Preenchido durante a fase de calibração/treinamento
    }

def send_data_to_api(payload):
    """
    Envia os dados empacotados para a API Flask via HTTP POST.
    """
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 201:
            print("Sucesso! Dados enviados para o banco:", response.json())
        else:
            print(f"Erro ao enviar dados. Status: {response.status_code}, Resposta: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão com a API: {e}")

if __name__ == "__main__":
    print("Iniciando leitura do sensor AS7265x...")
    
    # Loop de leitura contínua (exemplo: envia a cada 10 segundos)
    try:
        while True:
            dados_leitura = read_sensor_data()
            print(f"Enviando leitura de Glicose: {dados_leitura['glucose_mg_dl']} mg/dL")
            send_data_to_api(dados_leitura)
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("Programa encerrado pelo usuário.")