"""
URL configuration for webapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from main.views import login_view, monitoring_view,configuracion_view,camera_stream,calibrar_camara_paso_final,calibrar_camara_paso1,calibrar_camara_paso2,calibrar_camara_paso3,configurar_deteccion,iniciar_deteccion,stream_video,actualizar_poligono,etiquetar_poligono,agregar_poligono,quitar_poligono,obtener_detecciones,obtener_info_mesas,iniciar_tracking,detectar_mesas

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', login_view, name='login'),
    path('monitoreo/', monitoring_view, name='monitoreo'),
    path('configuracion/', configuracion_view, name='configuracion'),
    path('logout/', login_view, name='logout'),
    #path('configuracion/calibrar/',calibrar_camara, name='calibrar_camara'),
    path('camera_stream/', camera_stream, name='camera_stream'),
    path('configuracion/calibrar/paso1/', calibrar_camara_paso1, name='calibrar_camara_paso1'),
    path('configuracion/calibrar/paso2/', calibrar_camara_paso2, name='calibrar_camara_paso2'),
    path('configuracion/calibrar/paso3/', calibrar_camara_paso3, name='calibrar_camara_paso3'),
    path('configuracion/calibrar/final/', calibrar_camara_paso_final, name='calibrar_camara_paso_final'),
    path('configuracion/configurar-deteccion/', configurar_deteccion, name='configurar_deteccion'),
    path('configuracion/iniciar-deteccion/', iniciar_deteccion, name='iniciar_deteccion'),
    path('configuracion/stream-video/', stream_video, name='stream_video'),
    path('configuracion/actualizar-poligono/', actualizar_poligono, name='actualizar_poligono'),
    path('configuracion/etiquetar-poligono/', etiquetar_poligono, name='etiquetar_poligono'),
    path('configuracion/agregar-poligono/', agregar_poligono, name='agregar_poligono'),
    path('configuracion/quitar-poligono/', quitar_poligono, name='quitar_poligono'),
    path('configuracion/obtener-detecciones/',obtener_detecciones, name='obtener_detecciones'),
    path('configuracion/obtener_info_mesas/',obtener_info_mesas, name='obtener_info_mesas'),
    path('configuracion/iniciar-tracking/', iniciar_tracking, name='iniciar_tracking'),
    path('configuracion/detectar-mesas/', detectar_mesas, name='detectar_mesas'),

]