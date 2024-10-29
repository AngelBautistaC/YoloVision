import os
import firebase_admin
from firebase_admin import credentials, db

# Obtener la ruta de la carpeta donde se encuentra firebase_config.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Combinar la ruta de la carpeta con el nombre del archivo JSON
json_path = os.path.join(base_dir, "visionpucp-53816-firebase-adminsdk-4uyge-819a9c7e7c.json")

# Cargar las credenciales usando la ruta relativa
cred = credentials.Certificate(json_path)

# Inicializar la app de Firebase con la URL de la base de datos
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://visionpucp-53816-default-rtdb.firebaseio.com'
})
