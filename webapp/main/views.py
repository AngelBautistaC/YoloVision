from django.shortcuts import render, redirect

def login_view(request):
    if request.method == 'POST':
        # Aquí iría la lógica de autenticación en el futuro
        return redirect('monitoreo')
    return render(request, 'login.html')

def monitoring_view(request):
    return render(request, 'monitoreo.html')