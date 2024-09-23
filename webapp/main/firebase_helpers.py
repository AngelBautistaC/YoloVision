# firebase_helpers.py
from firebase_admin import db

def get_camera_by_ip(ip_url):
    cameras_ref = db.reference('cameras')
    cameras = cameras_ref.get()
    if cameras:
        for cam_uuid, cam_data in cameras.items():
            if cam_data.get('ip_url') == ip_url:
                return cam_uuid, cam_data
    return None, None

def get_camera_by_uuid(camera_uuid):
    camera_ref = db.reference(f'cameras/{camera_uuid}')
    cam_data = camera_ref.get()
    if cam_data:
        return cam_data
    return None

def save_camera_data(camera_uuid, data):
    camera_ref = db.reference(f'cameras/{camera_uuid}')
    camera_ref.set(data)
