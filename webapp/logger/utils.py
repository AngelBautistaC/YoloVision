from .models import LogEntry
from django.contrib.auth import get_user_model
from django.http import HttpRequest

def create_log_entry(request: HttpRequest, action: str, details: str = None):
    """
    Crea una nueva entrada de log en la base de datos.

    Args:
        request: La solicitud HTTP actual.
        action: La acción realizada.
        details: Detalles adicionales de la acción.
    """
    user_identifier = request.session.get('user_email')  # Obtener el email de la sesión

    ip_address = request.META.get('REMOTE_ADDR')

    LogEntry.objects.create(
        user=user_identifier,
        action=action,
        details=details,
        ip_address=ip_address
    )