from flask import Blueprint, jsonify
from services.supabase_client import supabase

glucose_routes = Blueprint("glucose_routes", __name__)

@glucose_routes.route("/glucose", methods=["GET"])
def get_glucose():

    response = supabase.table("glucose_data").select("*").execute()

    return jsonify(response.data)