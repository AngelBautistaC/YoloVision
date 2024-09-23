# firebase_config.py
import firebase_admin
from firebase_admin import credentials, db

# Path to your service account key JSON file
cred = credentials.Certificate("/Users/angbauti/Desktop/YoloVision/webapp/main/visionpucp-53816-firebase-adminsdk-4uyge-819a9c7e7c.json")

# Initialize the Firebase app with the service account and database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://visionpucp-53816-default-rtdb.firebaseio.com'
})
