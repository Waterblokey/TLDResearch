from django.urls import path
from . import views
from .SignOn import signinuser
from .SignOn import createuser

urlpatterns = [
    path("",views.home, name='home'),
    path('summarize/', views.handle_pdf_upload, name='upload_pdf'),
    path('login/', signinuser.signIn, name="login"),
    path('signup/', createuser.create_user_view, name='signup')
]