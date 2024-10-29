# firebase_config.py
import firebase_admin
from firebase_admin import credentials, firestore, auth

<<<<<<< Updated upstream
# Path to your service account key JSON file
cred = credentials.Certificate("/Users/angbauti/Desktop/YoloVision/webapp/main/visionpucp-53816-firebase-adminsdk-4uyge-819a9c7e7c.json")

# Initialize the Firebase app with the service account and database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://visionpucp-53816-default-rtdb.firebaseio.com'
})
=======
# Get the path of the Firebase Admin SDK JSON file
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "visionpucp-53816-firebase-adminsdk-4uyge-819a9c7e7c.json")

# Load credentials and initialize the Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate(json_path)
    firebase_admin.initialize_app(cred)

# Initialize Firestore database
firestore_db = firestore.client()
>>>>>>> Stashed changes
