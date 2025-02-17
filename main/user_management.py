# user_management.py
from firebase_admin import auth, db
import uuid

def create_user(email, password, display_name, role):
    # Create user in Firebase Authentication
    user_record = auth.create_user(
        email=email,
        password=password,
        display_name=display_name,
    )

    # Generate a unique UUID for the user
    user_id = user_record.uid

    # Save additional data (like role) in Firebase Database
    ref = db.reference(f'/users/{user_id}')
    ref.set({
        'display_name': display_name,
        'email': email,
        'role': role
    })

    return user_id

# Example usage
user_id = create_user(
    email="user@example.com",
    password="UserPassword123",
    display_name="John Doe",
    role="operator"
)
print(f"User created with UID: {user_id}")
