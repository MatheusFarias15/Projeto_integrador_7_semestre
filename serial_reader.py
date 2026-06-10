import json
import time
import os
import requests
import serial
from serial.tools import list_ports


def get_default_serial_port() -> str:
    """Tenta detectar automaticamente a porta USB do ESP32."""
    env_port = os.getenv("SERIAL_PORT")
    if env_port:
        return env_port

    ports = list_ports.comports()
    for port in ports:
        desc = (port.description or "").lower()
        if "ch9102" in desc or "usb-enhanced-serial" in desc or "cp210" in desc or "ft232" in desc:
            return port.device

    return "COM3"


# Configurações padrão - ajuste conforme a sua porta serial e servidor
SERIAL_PORT = get_default_serial_port()
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/glucose")
BACKEND_SENSOR_URL = os.getenv("BACKEND_SENSOR_URL", "http://localhost:5000/sensor-reading")
DEBUG_SERIAL = os.getenv("SERIAL_DEBUG", "0") == "1"
NO_DATA_TIMEOUT = float(os.getenv("SERIAL_NO_DATA_TIMEOUT", "5.0"))


def list_serial_ports() -> None:
    """Mostra portas seriais disponíveis no sistema."""
    ports = list_ports.comports()
    if not ports:
        print("Nenhuma porta serial encontrada.")
        return

    print("Portas seriais disponíveis:")
    for port in ports:
        print(f"  - {port.device}: {port.description}")


def parse_line(line: str) -> tuple[str, str] | None:
    """Parse uma linha no formato >CHAVE:VALOR"""
    line = line.strip()
    if not line.startswith(">") or ":" not in line:
        return None

    payload = line[1:].split(":", 1)
    if len(payload) != 2:
        return None

    key, value = payload[0].strip(), payload[1].strip()
    return key, value


def build_sensor_payload(sensor_data: dict) -> dict:
    """
    Cria o JSON para enviar a um novo endpoint /sensor-reading (para IA).
    Mapeia chaves do ESP32 para campos esperados.
    """
    # Mapeamento de nomes do ESP32 para nomes esperados
    mapping = {
        'BPM': 'bpm',
        'DC_IR': 'dc_ir',
        'AC_IR_Limpo': 'ac_ir',
        'Transmitancia_DC': 'transmitancia_dc',
        'Transmitancia_AC': 'transmitancia_ac',
    }
    
    payload = {}
    for esp32_key, api_key in mapping.items():
        if esp32_key in sensor_data:
            try:
                payload[api_key] = float(sensor_data[esp32_key])
            except (ValueError, TypeError):
                payload[api_key] = None
    
    return payload


def build_payload(sensor_data: dict) -> dict:
    """Cria o JSON que será enviado para a API Flask (endpoint legacy /glucose)."""
    return {
        "glucose_mg_dl": None,
        "spectral_transmittance_data": json.dumps(sensor_data),
        "box_temperature_celsius": None,
        "measurement_phase": "arduino",
        "real_concentration": None,
        "volunteer_notes": "Dados lidos via serial do Arduino",
    }


def send_to_backend(sensor_data: dict) -> None:
    """Envia dados para endpoint legado /glucose."""
    payload = build_payload(sensor_data)
    response = requests.post(BACKEND_URL, json=payload, timeout=10)
    response.raise_for_status()
    print("Dados enviados com sucesso:", sensor_data)


def send_sensor_reading(sensor_data: dict) -> None:
    """Envia dados de sensor para novo endpoint /sensor-reading (IA)."""
    payload = build_sensor_payload(sensor_data)
    response = requests.post(BACKEND_SENSOR_URL, json=payload, timeout=10)
    response.raise_for_status()
    if DEBUG_SERIAL:
        print("Leitura de sensor enviada com sucesso:", payload)


def main() -> None:
    print(f"Abrindo porta serial: {SERIAL_PORT} @ {SERIAL_BAUDRATE}")
    print(f"Enviando leituras legadas para: {BACKEND_URL}")
    print(f"Enviando leituras de sensor para: {BACKEND_SENSOR_URL}")
    list_serial_ports()

    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1) as ser:
            current_data: dict[str, float] = {}
            last_line_time = time.monotonic()

            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                now = time.monotonic()

                if line:
                    last_line_time = now

                if DEBUG_SERIAL and line:
                    print(f"DEBUG: raw_line={line!r}")

                if not line:
                    if now - last_line_time > NO_DATA_TIMEOUT:
                        print(f"AVISO: nenhum dado recebido em {NO_DATA_TIMEOUT} segundos.")
                        last_line_time = now
                    continue

                parsed = parse_line(line)
                if parsed is None:
                    if DEBUG_SERIAL:
                        print(f"DEBUG: linha ignorada={line!r}")
                    continue

                key, value = parsed
                try:
                    current_data[key] = float(value)
                except ValueError:
                    current_data[key] = value

                if DEBUG_SERIAL:
                    print(f"DEBUG: current_data={current_data}")

                # Envia quando a última linha esperada chegar
                if key == "Transmitancia_AC":
                    try:
                        # Envia para novo endpoint /sensor-reading (IA)
                        send_sensor_reading(current_data)
                    except Exception as err:
                        print(f"Erro ao enviar dados de sensor: {err}")
                    finally:
                        current_data = {}

                time.sleep(0.01)

    except serial.SerialException as err:
        print(f"Erro na porta serial: {err}")
    except KeyboardInterrupt:
        print("\nLeitura serial interrompida pelo usuário.")


if __name__ == "__main__":
    main()
