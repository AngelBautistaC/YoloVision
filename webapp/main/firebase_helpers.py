    # firebase_helpers.py
from firebase_admin import db

def get_camera_by_ip(ip_url):
    # Referencia a la raíz de los ambientes
    ref = db.reference('ambientes')
    snapshot = ref.get()

    # Recorrer todos los ambientes
    for ambiente_nombre, ambiente_data in snapshot.items():
        # Recorrer los pisos dentro de cada ambiente
        for piso, piso_data in ambiente_data.get('piso', {}).items():
            # Recorrer las cámaras dentro de cada piso
            for cam_uuid, cam_data in piso_data.get('camara', {}).items():
                # Verificar si la URL IP coincide
                if cam_data.get('ip_url') == ip_url:
                    # Retornar el UUID de la cámara y los datos de la cámara
                    return cam_uuid, cam_data
    # Si no se encontró la cámara, retornar None, None
    return None, None


def get_camera_by_uuid(camera_uuid): 
    # Convertir el UUID a string si es necesario
    camera_uuid_str = str(camera_uuid)

    # Referencia a la raíz de los ambientes
    ref = db.reference('ambientes')
    snapshot = ref.get()

    # Recorrer todos los ambientes
    for ambiente_nombre, ambiente_data in snapshot.items():
        print(f"Ambiente: {ambiente_nombre}")
        # Recorrer los pisos dentro de cada ambiente
        for piso, piso_data in ambiente_data.get('piso', {}).items():
            print(f"  Piso: {piso}")
            # Recorrer las cámaras dentro de cada piso
            for cam_uuid, cam_data in piso_data.get('camara', {}).items():
                print(f"    Cámara mmmm: {camera_uuid_str}")
                print(f"    Cámara UUID: {cam_uuid}")

                # Comparar los UUIDs como cadenas
                if cam_uuid == camera_uuid_str:
                    print(f"    Entroooooo")
                    return cam_data
    # Si no se encontró la cámara, retornar None
    return None







def save_camera_data(ambiente_nombre, piso, camera_uuid, camera_data):
 
    camera_ref = db.reference(f"ambientes/{ambiente_nombre}/piso/{piso}/camara/{camera_uuid}")
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

    print(f"Cámara guardada correctamente en {ambiente_nombre} - Piso: {piso}")



def save_camera_data2(camera_uuid, camera_data):
    # Obtener referencia a la raíz de los ambientes
    # Referencia a la ubicación específica en Firebase
    camera_ref = db.reference(f'cameras/{camera_uuid}')
    
    # Guardar los datos de la cámara
    camera_ref.set(camera_data)

