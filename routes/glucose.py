from flask import Blueprint, jsonify, request
from services.supabase_client import supabase

glucose_routes = Blueprint("glucose_routes", __name__)

# Rota GET existente para ler os dados
@glucose_routes.route("/glucose", methods=["GET"])
def get_glucose():
    response = supabase.table("glucose_data").select("*").execute()
    return jsonify(response.data), 200

# Nova Rota POST para receber dados do Raspberry Pi
@glucose_routes.route("/glucose", methods=["POST"])
def add_glucose_reading():
    try:
        # Pega os dados enviados pelo Raspberry Pi no formato JSON
        data = request.get_json()

        # Estrutura esperada do payload
        new_reading = {
            "glucose_mg_dl": data.get("glucose_mg_dl"),
            "spectral_transmittance_data": data.get("spectral_transmittance_data"),
            "box_temperature_celsius": data.get("box_temperature_celsius"),
            "measurement_phase": data.get("measurement_phase"),
            "real_concentration": data.get("real_concentration", None),
            "volunteer_notes": data.get("volunteer_notes", "")
        }

        # Insere no Supabase
        response = supabase.table("glucose_data").insert(new_reading).execute()
        
        return jsonify({"message": "Leitura salva com sucesso!", "data": response.data}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500