

from django.apps import AppConfig

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        # Import and initialize Firebase here
        import sys
        if 'runserver' in sys.argv or 'celery' in sys.argv or 'celery.exe' in sys.argv:
            from . import firebase_config
