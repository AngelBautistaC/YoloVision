# celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set the default Django settings module for the 'celery' command-line program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webapp.settings')  # Replace with your project name

app = Celery('webapp')

# Import firebase_config to initialize Firebase Admin SDK
from main import firebase_config  # Ensure this is the correct path to firebase_config.py

# Using a string here means the worker will not have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

from celery.schedules import crontab

app.conf.beat_schedule = {
    'check-connection-every-minute': {
        'task': 'main.tasks.check_connection_task',
        'schedule': 60.0,  # cada 60 seg chequea conexión
    },
    'run-detection-regularly': {
        'task': 'main.tasks.run_detection_for_configured_cameras',
        'schedule': 5.0, # cada 5 seg intenta detección en las cámaras en stable si ya pasó su detection_interval
    },
}