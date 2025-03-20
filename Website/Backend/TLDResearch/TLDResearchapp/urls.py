from django.urls import path
from . import views
from .SignOn import createuser
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("",views.home, name='home'),
    path('main/', views.main, name="main"),
    path('summarize/', views.handle_pdf_upload, name='upload_pdf'),
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page="login"), name="logout"),
    path('signup/', createuser.signup, name='signup'),
    path('summaries/', views.get_summaries, name='get-summaries')
]
