from django.shortcuts import redirect
from django.urls import reverse
from django.contrib import messages
from django.http import HttpResponseRedirect
from firebase_admin import db
from firebase_admin import auth, firestore

class CalibrationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Lista de URLs permitidas en el flujo de calibración
        calibration_urls = [
            '/configuracion/calibrar/paso1/',
            '/configuracion/calibrar/paso2/',
            '/configuracion/calibrar/paso3/',
        ]

        # Comprueba si estamos en una página de calibración
        is_calibration_page = any(request.path.startswith(url) for url in calibration_urls)

        # Si no estamos en una página de calibración pero hay datos de calibración en la sesión
        if not is_calibration_page and any(key.startswith('calibration_') for key in request.session.keys()):
            # Limpia los datos de calibración de la sesión
            keys_to_remove = [key for key in request.session.keys() if key.startswith('calibration_')]
            for key in keys_to_remove:
                del request.session[key]

        return response




class RoleMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if request.user.is_authenticated:
            # Obtener el UID del usuario autenticado
            try:
                decoded_token = auth.verify_id_token(request.COOKIES.get('firebase_token')) # Suponiendo que guardas el token de Firebase en una cookie
                uid = decoded_token['uid']

                # Autenticar al usuario con el UID
                user = auth.get_user(uid)

                # Forzar la actualización de la sesión con el rol desde Firebase
                firestore_client = firestore.client()
                user_data = firestore_client.collection('users').document(user.uid).get()
                if user_data.exists:
                    role = user_data.to_dict().get('role', 'user')
                    request.session['user_role'] = role
                    request.session.modified = True  # Forzar actualización de la sesión
                    print(f"RoleMiddleware: Usuario {request.user.username}, Rol actualizado en sesión: {role}")  # <-- Depuración
                else:
                    print(f"RoleMiddleware: No se encontró el rol para el usuario {request.user.username}")

            except auth.InvalidIdTokenError:
                print("RoleMiddleware: Token de Firebase inválido.")
            except auth.UserNotFoundError:
                print("RoleMiddleware: Usuario no encontrado en Firebase.")
            except Exception as e:
                print(f"RoleMiddleware: Error al obtener el rol: {e}")

        return response