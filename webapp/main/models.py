from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User

def create_roles():
    # Create groups
    user_group, created = Group.objects.get_or_create(name='user')
    operator_group, created = Group.objects.get_or_create(name='operator')
    admin_group, created = Group.objects.get_or_create(name='administrator')
    
    # Set permissions for each role
    # Example permissions - customize based on your models and requirements
    user_permissions = ['view_data']
    operator_permissions = ['view_data', 'change_data']
    admin_permissions = ['view_data', 'change_data', 'delete_data', 'add_data']

    # Assign permissions to groups
    for perm in user_permissions:
        permission = Permission.objects.get(codename=perm)
        user_group.permissions.add(permission)

    for perm in operator_permissions:
        permission = Permission.objects.get(codename=perm)
        operator_group.permissions.add(permission)

    for perm in admin_permissions:
        permission = Permission.objects.get(codename=perm)
        admin_group.permissions.add(permission)

# Run create_roles() once to set up the roles
