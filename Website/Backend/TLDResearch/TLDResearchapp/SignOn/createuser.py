from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def create_user_view(request):
    if request.method == 'POST':
        print("Raw request body:", request.body)  # Debug print
        data = json.loads(request.body.decode("utf-8"))  # Parse JSON body
        username = data.get("username")
        password = data.get("password")
        print((username, password))
        if not all([username, password]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        try:
            user = User.objects.create_user(username=username, password=password)
            return JsonResponse({'message': 'User created successfully'}, status=201)
        except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)