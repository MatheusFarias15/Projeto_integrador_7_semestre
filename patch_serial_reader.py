from pathlib import Path

p = Path('serial_reader.py')
t = p.read_text(encoding='utf-8')
old = '''SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/glucose")


def parse_line(line: str) -> tuple[str, str] | None:
'''
new = '''SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/glucose")
DEBUG_SERIAL = os.getenv("SERIAL_DEBUG", "0") == "1"


def parse_line(line: str) -> tuple[str, str] | None:
'''
if old not in t:
    raise SystemExit('Old block not found')
t = t.replace(old, new, 1)
old2 = '''            if key == "Transmitancia_AC":
                try:
                    send_to_backend(current_data)
                except Exception as err:
                    print(f"Erro ao enviar dados: {err}")
                finally:
                    current_data = {}

            time.sleep(0.01)
'''
new2 = '''            if key == "Transmitancia_AC":
                try:
                    send_to_backend(current_data)
                except Exception as err:
                    print(f"Erro ao enviar dados: {err}")
                finally:
                    current_data = {}

            if DEBUG_SERIAL:
                print(f"DEBUG: current_data={current_data}")

            time.sleep(0.01)
'''
if old2 not in t:
    raise SystemExit('Old block2 not found')
t = t.replace(old2, new2, 1)
old3 = '''    with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1) as ser:
        current_data: dict[str, float] = {}

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
'''
new3 = '''    with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1) as ser:
        current_data: dict[str, float] = {}
        last_line_time = time.monotonic()

        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if DEBUG_SERIAL and line:
                print(f"DEBUG: raw_line={line!r}")
'''
if old3 not in t:
    raise SystemExit('Old block3 not found')
t = t.replace(old3, new3, 1)
p.write_text(t, encoding='utf-8')
print('patched')
