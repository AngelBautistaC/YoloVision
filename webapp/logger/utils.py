from .models import LogEntry
from django.contrib.auth import get_user_model
from django.http import HttpRequest
import datetime

def create_log_entry(request: HttpRequest, action: str, details: str = None):
    """
    Crea una nueva entrada de log en la base de datos desde una vista.

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

def create_celery_log_entry(action: str, details: str = None):
    """
    Crea una nueva entrada de log en la base de datos desde una tarea de Celery.

    Args:
        action: La acción realizada.
        details: Detalles adicionales de la acción.
    """
    LogEntry.objects.create(
        user="Sistema",  # Usamos "System" para logs generados por el sistema
        action=action,
        details=details,
        ip_address="N/A"  # No hay IP disponible en una tarea de Celery
    )