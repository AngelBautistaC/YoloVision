from django.shortcuts import render
from .models import LogEntry

def log_view(request):
    logs = LogEntry.objects.all().order_by('-timestamp')
    return render(request, 'logger/log_view.html', {'logs': logs})