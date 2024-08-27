from django.shortcuts import render, redirect

def login_view(request):
    if request.method == 'POST':
        # Aquí iría la lógica de autenticación en el futuro
        return redirect('dashboard')
    return render(request, 'login.html')

def dashboard_view(request):
    return render(request, 'dashboard.html')