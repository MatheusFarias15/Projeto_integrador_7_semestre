#include <Arduino.h>
#include <Wire.h>
#include <SPI.h> 
#include "MAX30105.h"
#include <Adafruit_ADS1X15.h> // A biblioteca que você adicionou no .ini

MAX30105 particleSensor;
Adafruit_ADS1115 ads; // Inicializa o ADC de precisão (Endereço I2C padrão 0x48)

// Pino de disparo do Canhão Infravermelho (LED 3W)
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

  // Trava de segurança: LED 3W começa desligado!
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

  // Configuração do ganho do ADS1115 (Lê até +/- 6.144V)
  // Perfeito para o sinal de até 3.3V que virá do MCP6002
  ads.setGain(GAIN_TWOTHIRDS); 

  // Configuração do MAX30105
  byte ledBrightness = 0x1F; 
  byte sampleAverage = 4;
  byte ledMode = 2; 
  byte sampleRate = 400; 
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}

void loop() {
  uint32_t rawRed = particleSensor.getRed(); 
  uint32_t rawIR = particleSensor.getIR(); 

  // ==========================================
  // TRAVA DE AR (Dedo fora do sensor)
  // ==========================================
  if (rawRed < 10000) {
    digitalWrite(pinoMosfet, LOW); // DESLIGA o LED de 3W para não torrar atoa
    pulsoDetectado = false;
    subindo = false;
    bpmMedio = 0;
    return;
  }

  // O dedo está no sensor! LIGA o LED de 3W para atravessar o dedo
  digitalWrite(pinoMosfet, HIGH);

  // ==========================================
  // LEITURA DO FOTODIODO BPW34 (Via ADS1115 Pino A0)
  // ==========================================
  int16_t rawGlicose = ads.readADC_SingleEnded(0);

  // ==========================================
  // FILTRAGEM DO CANAL DE GLICOSE
  // ==========================================
  // Assim como o MAX, a luz do LED 3W que passa pelo dedo tem uma 
  // parcela estática (osso/pele) e uma parcela pulsante (sangue).
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

      // Variável de saída para a IA e Teleplot
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
  Serial.print(">DC_IR:"); Serial.println(dcIR);
  Serial.print(">AC_IR_Limpo:"); Serial.println(smoothACIR);
  
  // As duas novas features vitais para o modelo PLS estimar a glicose:
  Serial.print(">Transmitancia_DC:"); Serial.println(dcGlicose);
  Serial.print(">Transmitancia_AC:"); Serial.println(smoothACGlicose);
}