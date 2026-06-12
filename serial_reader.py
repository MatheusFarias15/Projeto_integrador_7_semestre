import json
import math
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
    mapping = {
        'BPM': 'bpm',
        'SpO2': 'spo2',
        'SPO2': 'spo2',
        'DC_IR': 'dc_ir',
        'AC_IR_Limpo': 'ac_ir',
        'IR_MAX30102': 'ir_max30102',
        'MAX30102_IR': 'ir_max30102',
        'RED_MAX30102': 'red_max30102',
        'MAX30102_RED': 'red_max30102',
        'Transmitancia_DC': 'transmitancia_dc',
        'Transmitancia_AC': 'transmitancia_ac',
        'BPW34_RAW': 'bpw34_raw',
        'BPW34_Raw': 'bpw34_raw',
        'BPW34_VOLTAGE': 'bpw34_voltage',
        'BPW34_Voltage': 'bpw34_voltage',
        'BPW34_CURRENT': 'bpw34_current',
        'BPW34_AC': 'bpw34_ac',
        'BPW34_DC': 'bpw34_dc',
        'BPW34_RMS': 'bpw34_rms',
        'BPW34_PEAK': 'bpw34_peak',
        'BPW34_MEAN': 'bpw34_mean',
        'IR_940': 'ir_940_intensity',
        'IR_940_INTENSITY': 'ir_940_intensity',
        'IR_940_TRANSMITTANCE': 'ir_940_transmittance',
        'RED_660': 'red_660',
        'Temperatura': 'temperatura',
        'TEMPERATURA': 'temperatura',
        'Temperature': 'temperatura',
    }
    
    payload = {}
    for esp32_key, api_key in mapping.items():
        if esp32_key in sensor_data:
            try:
                payload[api_key] = float(sensor_data[esp32_key])
            except (ValueError, TypeError):
                payload[api_key] = None

    received_light = (
        payload.get("ir_940_transmittance")
        or payload.get("bpw34_voltage")
        or payload.get("bpw34_dc")
        or payload.get("transmitancia_dc")
    )
    emitted_light = payload.get("ir_940_intensity") or payload.get("ir_max30102") or payload.get("dc_ir")
    if received_light is not None and emitted_light and emitted_light > 0:
        transmittance = received_light / emitted_light
        payload["transmittance"] = transmittance
        if transmittance > 0:
            payload["absorbance"] = -math.log10(transmittance)

    ac_component = payload.get("bpw34_ac") or payload.get("transmitancia_ac")
    dc_component = payload.get("bpw34_dc") or payload.get("transmitancia_dc")
    if ac_component is not None and dc_component:
        payload["pulsatile_index"] = ac_component / dc_component

    if payload.get("transmitancia_ac") is not None and payload.get("ac_ir"):
        payload["ratio_ir_trans"] = payload["transmitancia_ac"] / payload["ac_ir"]

    if payload.get("ac_ir") is not None and payload.get("dc_ir"):
        payload["ir_ratio"] = payload["ac_ir"] / payload["dc_ir"]

    if payload.get("ir_940_intensity") is not None and payload.get("bpw34_voltage"):
        payload["ratio_ir_bpw34"] = payload["ir_940_intensity"] / payload["bpw34_voltage"]
    
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


def is_empty_sensor_packet(sensor_data: dict) -> bool:
    """Retorna True quando o pacote contem apenas leituras numericas zeradas."""
    numeric_values = [
        value
        for value in sensor_data.values()
        if isinstance(value, (int, float))
    ]
    return bool(numeric_values) and all(value == 0 for value in numeric_values)


def send_to_backend(sensor_data: dict) -> None:
    """Envia dados para endpoint legado /glucose."""
    payload = build_payload(sensor_data)
    response = requests.post(BACKEND_URL, json=payload, timeout=10)
    response.raise_for_status()
    print("Dados enviados com sucesso:", sensor_data)


def send_sensor_reading(sensor_data: dict) -> None:
    """Envia dados de sensor para novo endpoint /sensor-reading (IA)."""
    if is_empty_sensor_packet(sensor_data):
        if DEBUG_SERIAL:
            print("DEBUG: pacote zerado ignorado")
        return

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
                if key in {"Transmitancia_AC", "BPW34_MEAN", "BPW34_Peak", "IR_940_TRANSMITTANCE"}:
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
