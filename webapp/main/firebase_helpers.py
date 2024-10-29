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
            # Recorrer las cámaras dentro de cada piso
            for cam_uuid, cam_data in piso_data.get('camara', {}).items():
                # Verificar si la URL IP coincide
                if cam_data.get('ip_url') == ip_url:
                    # Retornar el UUID de la cámara y los datos de la cámara
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
        print(f"Ambiente: {ambiente_nombre}")
        # Recorrer los pisos dentro de cada ambiente
        for piso, piso_data in ambiente_data.get('piso', {}).items():
            print(f"  Piso: {piso}")
            # Recorrer las cámaras dentro de cada piso
            for cam_uuid, cam_data in piso_data.get('camara', {}).items():
                print(f"    Cámara mmmm: {camera_uuid_str}")
                print(f"    Cámara UUID: {cam_uuid}")

                # Comparar los UUIDs como cadenas
                if cam_uuid == camera_uuid_str:
                    print(f"    Entroooooo")
                    return cam_data
    # Si no se encontró la cámara, retornar None
    return None


def save_camera_data(ambiente_nombre, piso, camera_uuid, camera_data):
 
    camera_ref = db.reference(f"ambientes/{ambiente_nombre}/piso/{piso}/camara/{camera_uuid}")
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

    print(f"Cámara guardada correctamente en {ambiente_nombre} - Piso: {piso}")



def save_camera_data2(camera_uuid, camera_data):
    # Obtener referencia a la raíz de los ambientes
    # Referencia a la ubicación específica en Firebase
    camera_ref = db.reference(f'cameras/{camera_uuid}')
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

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
        user_id = response_data["localId"]
        role = "user"  # Or pull this from Firestore if stored there
        return user_id, role

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