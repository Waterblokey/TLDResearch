from django.contrib.auth.models import User
from django.http import JsonResponse

def create_user_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')

        if not all([username, password, email]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        try:
            user = User.objects.create_user(username=username, password=password, email=email)
            return JsonResponse({'message': 'User created successfully'}, status=201)
        except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)