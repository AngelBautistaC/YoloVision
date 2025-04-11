#urls.py
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views # Importar las vistas de autenticación de Django
from main.views import actualizar_campo_poligono, estadisticas_view,store_line_points,get_monitoreo_data,logout_view,admin_logs,access_denied, eliminar_camaras,crear_form_view,crear_view,login_view, monitoring_view,configuracion_view,camera_stream,calibrar_camara_paso_final,calibrar_camara_paso1,calibrar_camara_paso3,configurar_deteccion,iniciar_deteccion,stream_video,actualizar_poligono,etiquetar_poligono,agregar_poligono,quitar_poligono,obtener_detecciones,obtener_info_mesas,iniciar_tracking,detectar_mesas,supervisar_camara,stream_supervisar_camara,obtener_info_mesas_uuid
urlpatterns = [
    path('logout/', logout_view, name='logout'),
    path('admin/', admin.site.urls),
    path('', login_view, name='login'),
    path('login/', login_view, name='login'),
    path('monitoreo/', monitoring_view, name='monitoreo'),
    path('monitoreo/data/', get_monitoreo_data, name='monitoreo_data'), # Add this line
    path('crear/', crear_view, name='crear'),
    path('crear_form/', crear_form_view, name='crear_form'),
    path('configuracion/', configuracion_view, name='configuracion'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'), # Usar la vista de logout de Django
    path('admin/logs/', admin_logs, name='admin_logs'),
    path('access_denied/', access_denied, name='access_denied'), 
    path('camera_stream/', camera_stream, name='camera_stream'),
    path('configuracion/calibrar/paso1/', calibrar_camara_paso1, name='calibrar_camara_paso1'),
    path('configuracion/calibrar/paso3/', calibrar_camara_paso3, name='calibrar_camara_paso3'),
    path('configuracion/calibrar/store_line_points/', store_line_points, name='store_line_points'),
    path('configuracion/calibrar/final/', calibrar_camara_paso_final, name='calibrar_camara_paso_final'),
    path('configuracion/configurar-deteccion/', configurar_deteccion, name='configurar_deteccion'),
    path('configuracion/iniciar-deteccion/', iniciar_deteccion, name='iniciar_deteccion'),
    path('configuracion/stream-video/', stream_video, name='stream_video'),
    path('configuracion/actualizar_campo_poligono/', actualizar_campo_poligono, name='actualizar_campo_poligono'),
    path('configuracion/actualizar-poligono/', actualizar_poligono, name='actualizar_poligono'),
    path('configuracion/etiquetar-poligono/', etiquetar_poligono, name='etiquetar_poligono'),
    path('configuracion/agregar-poligono/', agregar_poligono, name='agregar_poligono'),
    path('configuracion/quitar-poligono/', quitar_poligono, name='quitar_poligono'),
    path('configuracion/obtener-detecciones/',obtener_detecciones, name='obtener_detecciones'),
    path('configuracion/obtener_info_mesas/',obtener_info_mesas, name='obtener_info_mesas'),
    path('configuracion/iniciar-tracking/', iniciar_tracking, name='iniciar_tracking'),
    path('configuracion/detectar-mesas/', detectar_mesas, name='detectar_mesas'),
    path('monitoreo/supervisar-camara/<uuid:camera_uuid>/', supervisar_camara, name='supervisar_camara'),
    path('monitoreo/stream-supervisar-camara/<uuid:camera_uuid>/', stream_supervisar_camara, name='stream_supervisar_camara'),
    path('monitoreo/obtener_info_mesas/<uuid:camera_uuid>/', obtener_info_mesas_uuid, name='obtener_info_mesas_uuid'),
    path('monitoreo/eliminar_camaras/', eliminar_camaras, name='eliminar_camaras'),
     path('estadisticas/', estadisticas_view, name='estadisticas'), # Nueva URL
]