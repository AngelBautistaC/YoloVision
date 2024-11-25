from django.shortcuts import redirect

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


# Example of adding an is_admin attribute in a custom middleware
class RoleMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            # Assuming role is stored in Firebase Realtime Database under /users/<uid>/role
            user_ref = db.reference(f'/users/{request.user.username}/role')
            role = user_ref.get()
            request.user.is_admin = (role == 'admin')
        return self.get_response(request)
