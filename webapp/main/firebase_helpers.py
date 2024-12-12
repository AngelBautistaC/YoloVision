    # firebase_helpers.py
import firebase_admin
from firebase_admin import db
from firebase_admin import auth, firestore
from firebase_admin.exceptions import FirebaseError
from django.contrib.auth import authenticate, login
from django.core.exceptions import PermissionDenied
from django.contrib.auth.models import User  # Local Django User for session handling


import firebase_admin
from firebase_admin import auth, db
from django.conf import settings
from . import firebase_config  # Ensure Firebase is initialized
import requests


# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)  # Path to your Firebase service account key
    firebase_admin.initialize_app(cred)

def get_camera_by_ip(ip_url):
    # Referencia a la raíz de los ambientes
    ref = db.reference('ambientes')
    snapshot = ref.get()

    # Recorrer todos los ambientes
    for ambiente_nombre, ambiente_data in snapshot.items():
        # Recorrer los pisos dentro de cada ambiente
        for piso, piso_data in ambiente_data.get('piso', {}).items():
            # Verificar si piso_data es un diccionario
            if not isinstance(piso_data, dict):
                continue  # Saltar este piso

            # Recorrer las cámaras dentro de cada piso
            camaras = piso_data.get('camara', {})
            if not isinstance(camaras, dict):
                continue

            for cam_uuid, cam_data in camaras.items():
                # Verificar si la URL IP coincide
                if cam_data.get('ip_url') == ip_url:
                    # Añadir ambiente y piso al cam_data
                    cam_data['ambiente'] = ambiente_nombre
                    cam_data['piso'] = piso
                    return cam_uuid, cam_data
    # Si no se encontró la cámara, retornar None, None
    return None, None

def get_camera_by_uuid(camera_uuid): 
    # Convertir el UUID a string si es necesario
    camera_uuid_str = str(camera_uuid)

    # Referencia a la raíz de los ambientes
    ref = db.reference('ambientes')
    snapshot = ref.get()

    # Recorrer todos los ambientes
    for ambiente_nombre, ambiente_data in snapshot.items():
        # Recorrer los pisos dentro de cada ambiente
        for piso_nombre, piso_data in ambiente_data.get('piso', {}).items():
            # Recorrer las cámaras dentro de cada piso
            for cam_uuid, cam_data in piso_data.get('camara', {}).items():
                # Comparar los UUIDs como cadenas
                if cam_uuid == camera_uuid_str:
                    # Añadir ambiente y piso al cam_data
                    cam_data['ambiente'] = ambiente_nombre
                    cam_data['piso'] = piso_nombre
                    return cam_data
    # Si no se encontró la cámara, retornar None
    return None

def save_camera_data(ambiente_nombre, piso, camera_uuid, camera_data):
 
    camera_ref = db.reference(f"ambientes/{ambiente_nombre}/piso/{piso}/camara/{camera_uuid}")
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

    print(f"Cámara guardada correctamente en {ambiente_nombre} - Piso: {piso}")




def save_camera_data2(ambiente_nombre, piso, camera_uuid, camera_data):
    camera_ref = db.reference(f"ambientes/{ambiente_nombre}/piso/{piso}/camara/{camera_uuid}")
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

    print(f"Cámara guardada correctamente en {ambiente_nombre} - Piso: {piso}")

from firebase_admin import auth, firestore

def create_user_in_firebase(username, email, password, role):
    try:
        # Step 1: Create user in Firebase Authentication
        print("Attempting to create user in Firebase Authentication...")
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=username
        )
        user_id = user_record.uid
        print(f"User created with UID: {user_id}")

        # Step 2: Save user details in Firestore
        print("Attempting to save user details to Firestore...")
        firestore_db = firestore.client()
        user_ref = firestore_db.collection('users').document(user_id)

        user_ref.set({
            'username': username,
            'email': email,
            'role': role,
            'uid': user_id
        })
        print("User details saved in Firestore successfully.")
        return user_id

    except auth.EmailAlreadyExistsError:
        print("Error: Email is already registered.")
    except auth.InvalidPasswordError:
        print("Error: Password is invalid or too weak.")
    except Exception as e:
        print(f"Error creating user in Firebase: {e}")
    return None

FIREBASE_API_KEY ="AIzaSyCL9vshBD4m6BktNo7ku47o3da29Qp6sRQ"
def authenticate_firebase_user(email, password):
    """
    Authenticates user using Firebase email and password authentication.
    """
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        
        if "error" in response_data:
            print(f"Firebase Authentication Error: {response_data['error']['message']}")
            return None, None
        
        # Assuming role is retrieved from Firebase's custom claims, or you can store it in Firestore
        user = auth.get_user_by_email(email)
        firestore_client = firestore.client()
        user_data = firestore_client.collection('users').document(user.uid).get()
        
        if user_data.exists:
            role = user_data.to_dict().get('role', 'user')  # Default to 'user' if no role
            return user, role
        return None, None

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None, None

def save_user_to_firebase(email, role):
    """
    Save a new user with a specific role in Firebase Realtime Database.
    """
    ref = db.reference('users')  # Point to the 'users' node in Firebase
    user_data = {
        'email': email,
        'role': role,
    }
    ref.push(user_data)  # Save user data with a unique ID

def get_ambientes_and_pisos_from_db():
    ref = db.reference('ambientes')
    ambientes_snapshot = ref.get()

    ambientes_data = []
    if ambientes_snapshot:
        for ambiente_nombre, ambiente_data in ambientes_snapshot.items():
            pisos_dict = ambiente_data.get('piso', {})
            # Verificar si pisos_dict es un diccionario
            if not isinstance(pisos_dict, dict):
                print(f"Advertencia: Los datos de 'piso' en el ambiente '{ambiente_nombre}' no son un diccionario. Se omitirán estos datos.")
                continue  # Saltar este ambiente

            pisos = []
            for piso_nombre, piso_data in pisos_dict.items():
                if isinstance(piso_data, dict):
                    pisos.append(piso_nombre)
                else:
                    print(f"Advertencia: Los datos del piso '{piso_nombre}' en el ambiente '{ambiente_nombre}' no son un diccionario. Se omitirán este piso.")

            ambientes_data.append({
                'nombreAmbiente': ambiente_nombre,
                'pisos': pisos,
            })
    return ambientes_data

def delete_camera_from_firebase(camera_uuid):
    """
    Elimina una cámara y todas sus subentradas de Firebase, manteniendo las entradas padre.

    Args:
        camera_uuid (str): El UUID de la cámara a eliminar.
    """
    try:
         # Convertir el UUID a string si es necesario
        camera_uuid_str = str(camera_uuid)
         # Referencia a la raíz de los ambientes
        ref = db.reference('ambientes')
        snapshot = ref.get()
        # Recorrer todos los ambientes
        for ambiente_nombre, ambiente_data in snapshot.items():
            # Recorrer los pisos dentro de cada ambiente
            for piso_nombre, piso_data in ambiente_data.get('piso', {}).items():
                # Recorrer las cámaras dentro de cada piso
                if isinstance(piso_data, dict):
                    camaras = piso_data.get('camara', {})
                    if isinstance(camaras, dict):
                            for cam_uuid, cam_data in camaras.items():
                            # Comparar los UUIDs como cadenas
                                if cam_uuid == camera_uuid_str:
                                    camera_ref = db.reference(f"ambientes/{ambiente_nombre}/piso/{piso_nombre}/camara/{camera_uuid_str}")
                                    camera_ref.delete()
                                    print(f"Cámara con UUID '{camera_uuid_str}' eliminada correctamente de Firebase.")
                                    return  # Termina la función después de eliminar la cámara
        print(f"No se encontró cámara con UUID '{camera_uuid_str}' para eliminar en Firebase.")

    except FirebaseError as e:
        print(f"Error al eliminar la cámara con UUID '{camera_uuid}': {e}")
    except Exception as e:
        print(f"Error inesperado al eliminar la cámara con UUID '{camera_uuid}': {e}")