from django.shortcuts import render, redirect
import cv2



from django.http import StreamingHttpResponse, HttpResponseBadRequest,JsonResponse
import urllib.request
from django.shortcuts import render
import numpy as np


def login_view(request):
    if request.method == 'POST':
        # Aquí iría la lógica de autenticación en el futuro
        return redirect('monitoreo')
    return render(request, 'login.html')

def monitoring_view(request):
    return render(request, 'monitoreo.html')

def configuracion_view(request):
    return render(request, 'configuracion.html')



def calibrar_camara(request):
    step = request.GET.get('step', '1')
    context = {'step': step, 'camera_connected': False}

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        ip_port = request.POST.get('ip_port')
        try:
            # Intentar leer una imagen del stream MJPEG
            stream = urllib.request.urlopen(ip_port, timeout=10)
            bytes_data = b''
            for i in range(30):  # Leer algunos frames para asegurarnos
                bytes_data += stream.read(1024)
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        context['camera_connected'] = True
                        return JsonResponse({'success': True, 'message': 'Conexión exitosa'})
            
            return JsonResponse({'success': False, 'message': 'No se pudo leer el frame de la cámara después de 30 intentos'})
        except urllib.error.URLError as e:
            return JsonResponse({'success': False, 'message': f'Error de URL: {str(e)}'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'Error inesperado: {str(e)}'})

    return render(request, 'calibrar_camara.html', context)


def gen_frames(camera_url):
    stream = urllib.request.urlopen(camera_url)
    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def camera_stream(request):
    ip_port = request.GET.get('ip_port')
    if not ip_port:
        return HttpResponseBadRequest("IP:Port parameter is missing")
    return StreamingHttpResponse(gen_frames(ip_port),
                                 content_type='multipart/x-mixed-replace; boundary=frame')