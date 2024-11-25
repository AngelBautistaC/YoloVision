import os
import firebase_admin
from firebase_admin import credentials, firestore, auth, db
import os

# Obtén la ruta del archivo JSON del SDK de administrador de Firebase
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "visionpucp-53816-firebase-adminsdk-4uyge-819a9c7e7c.json")

# Inicializa la aplicación de Firebase si aún no está inicializada
if not firebase_admin._apps:
    cred = credentials.Certificate(json_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://visionpucp-53816-default-rtdb.firebaseio.com'  # Asegúrate de que la URL sea correcta
    })

# Inicializa Firestore y Realtime Database
firestore_db = firestore.client()
