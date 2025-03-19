from django.contrib.auth.models import User
from django.http import HttpResponseBadRequest
from django.http import JsonResponse

from django.contrib.auth import authenticate
import uuid
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def signIn(request):
    if request.method == 'POST':
        print("Raw request body:", request.body)  # Debug print
        data = json.loads(request.body.decode("utf-8"))  # Parse JSON body
        username = data.get("username")
        password = data.get("password")
        if not all([username, password]):
            return HttpResponseBadRequest('Bad call to backend api')
        user = authenticate(username=username,password=password)
        if user is not None:
            cookie_value = uuid.uuid4()
            request.session[cookie_value.hex] = username
            return JsonResponse({'session_cookie':cookie_value.hex})
    else:
        return HttpResponseBadRequest("Bad call to backend api")