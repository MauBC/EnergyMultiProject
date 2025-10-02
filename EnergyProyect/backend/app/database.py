from pymongo import MongoClient
import certifi
from .config import MONGO_URI

client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client["energy_proyect"]
