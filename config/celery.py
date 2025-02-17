# celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set the default Django settings module for the 'celery' command-line program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('config')  # Celery app instance named 'config'

# Import firebase_config to initialize Firebase Admin SDK
from main import firebase_config

app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

from celery.schedules import crontab

app.conf.beat_schedule = {
    'check-connection-every-minute': {
        'task': 'main.tasks.check_connection_task',
        'schedule': 60.0,
    },
    'run-detection-regularly': {
        'task': 'main.tasks.run_detection_for_configured_cameras',
        'schedule': 5.0,
    },
}