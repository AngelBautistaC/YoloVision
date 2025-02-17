from django.db import models
from django.conf import settings

class LogEntry(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=150, blank=True, null=True)  # Guardamos username o id
    action = models.CharField(max_length=255)
    details = models.TextField(blank=True, null=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)

    def __str__(self):
        return f"{self.timestamp} - {self.user} - {self.action}"