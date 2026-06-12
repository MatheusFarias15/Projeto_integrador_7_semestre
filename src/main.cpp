#include <Arduino.h>
#include <Wire.h>
#include <SPI.h> 
#include "MAX30105.h"
#include <Adafruit_ADS1X15.h> 

MAX30105 particleSensor;
Adafruit_ADS1115 ads; // Inicializa o ADC de precisão (Endereço I2C padrão 0x48)

// Pino de disparo do Emissor Infravermelho via MOSFET
const int pinoMosfet = 4;

// Variáveis de Filtro DC e AC (MAX30105)
float dcRed = 0;
float dcIR = 0;
float smoothACRed = 0;
float smoothACIR = 0;

// Variáveis de Filtro DC e AC (GLICOSE - BPW34)
float dcGlicose = 0;
float smoothACGlicose = 0;

// Variáveis do BPM
float ultimoACIR = 0;
bool subindo = false;
bool pulsoDetectado = false;
unsigned long ultimoRearme = 0;
unsigned long ultimoTempoBeat = 0;
float bpmMedio = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  // Configura o pino do MOSFET e inicia com o emissor desligado.
  pinMode(pinoMosfet, OUTPUT);
  digitalWrite(pinoMosfet, LOW);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Erro: MAX30105 mudo.");
    while (1);
  }

  if (!ads.begin()) {
    Serial.println("Erro: ADS1115 não encontrado no barramento I2C.");
    while (1);
  }

  // Configuração do ganho do ADS1115 (Ajuste conforme a sua necessidade de "zoom")
  ads.setGain(GAIN_TWOTHIRDS); 

  // Configuração do MAX30105
  byte ledBrightness = 0x1F; 
  byte sampleAverage = 4;
  byte ledMode = 2; 
  
  // CORREÇÃO: Usando 'int' para evitar o overflow do número 400
  int sampleRate = 400; 
  
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}

void loop() {
  uint32_t rawRed = particleSensor.getRed(); 
  uint32_t rawIR = particleSensor.getIR(); 

  // ==========================================
  // FILTRO DE DISCREPÂNCIA (Dedo ausente ou leitura morta)
  // ==========================================
  if (rawRed < 10000) {
    // Zera os filtros internos para evitar que o ruído contamine as médias móveis
    pulsoDetectado = false;
    subindo = false;
    bpmMedio = 0;
    dcIR = 0;
    dcGlicose = 0;
    smoothACIR = 0;
    smoothACGlicose = 0;

    digitalWrite(pinoMosfet, LOW);
    
    // Pula o resto dos cálculos matemáticos até o dedo voltar
    return; 
  }

  digitalWrite(pinoMosfet, HIGH);

  // ==========================================
  // LEITURA DO FOTODIODO BPW34 (Via ADS1115 Pino A0)
  // ==========================================
  int16_t rawGlicose = ads.readADC_SingleEnded(0);

  // ==========================================
  // FILTRAGEM DO CANAL DE GLICOSE
  // ==========================================
  // A variável inicializa baseada na leitura crua se estava zerada antes
  if (dcGlicose == 0) dcGlicose = rawGlicose;
  dcGlicose = (dcGlicose * 0.95) + (rawGlicose * 0.05);
  
  float acGlicose = rawGlicose - dcGlicose;
  smoothACGlicose = (smoothACGlicose * 0.8) + (acGlicose * 0.2);

  // ==========================================
  // CANAL DE REFERÊNCIA (MAX30105 - INFRAVERMELHO 880nm)
  // ==========================================
  if (dcIR == 0) dcIR = rawIR;
  dcIR = (dcIR * 0.95) + (rawIR * 0.05);
  float acIR = rawIR - dcIR;
  smoothACIR = (smoothACIR * 0.8) + (acIR * 0.2);

  // ==========================================
  // MÁQUINA DE ESTADOS: BPM
  // ==========================================
  if (smoothACIR > ultimoACIR) { subindo = true; }

  if (subindo && (smoothACIR < ultimoACIR) && !pulsoDetectado) {
    pulsoDetectado = true;
    subindo = false;
    
    long tempoAtual = millis();
    long deltaT = tempoAtual - ultimoTempoBeat;

    if (deltaT > 300 && deltaT < 2000) {
      float bpmInstantaneo = 60000.0 / (float)deltaT;
      if (bpmMedio == 0) bpmMedio = bpmInstantaneo; 
      bpmMedio = (bpmMedio * 0.90) + (bpmInstantaneo * 0.10);

      // Variável de saída do Batimento Cardíaco validado
      Serial.print(">BPM:"); Serial.println(bpmMedio);
    }
    
    ultimoTempoBeat = tempoAtual;
    ultimoRearme = tempoAtual;
  }

  if (pulsoDetectado && (millis() - ultimoRearme > 250)) { pulsoDetectado = false; }
  ultimoACIR = smoothACIR; 

  // ==========================================
  // SAÍDA DE DADOS (DATASET DA IA)
  // ==========================================
  Serial.print(">MAX30102_RED:"); Serial.println(rawRed);
  Serial.print(">MAX30102_IR:"); Serial.println(rawIR);
  Serial.print(">DC_IR:"); Serial.println(dcIR);
  Serial.print(">AC_IR_Limpo:"); Serial.println(smoothACIR);
  
  Serial.print(">BPW34_RAW:"); Serial.println(rawGlicose);
  Serial.print(">BPW34_DC:"); Serial.println(dcGlicose);
  Serial.print(">BPW34_AC:"); Serial.println(smoothACGlicose);
  Serial.print(">Transmitancia_DC:"); Serial.println(dcGlicose);
  
  // CORREÇÃO: Chave de fechamento adicionada após a última linha!
  Serial.print(">Transmitancia_AC:"); Serial.println(smoothACGlicose);
}
