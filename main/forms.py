# forms.py
from django import forms
from django.contrib.auth.models import User

class LoginForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'}))

class CreateUserForm(forms.Form):
    username = forms.CharField(label='Nombre de usuario', max_length=150, required=True)
    email = forms.EmailField(label='Correo electrónico', required=True)
    password = forms.CharField(label='Contraseña', widget=forms.PasswordInput, required=True)
    role = forms.ChoiceField(
        label='Rol',
        choices=[('user', 'Usuario'), ('operator', 'Operador'), ('admin', 'Administrador')],
        required=True
    )

