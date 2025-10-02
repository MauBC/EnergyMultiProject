from config import MONGO_URI
from pymongo import MongoClient
import certifi

client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client["energy_proyect"]

try:
    print("Colecciones en energy_proyect:", db.list_collection_names())
    test = db["energy_data"].find_one()
    print("Un documento de energy_data:", test)
except Exception as e:
    print("❌ Error conectando a MongoDB:", e)
