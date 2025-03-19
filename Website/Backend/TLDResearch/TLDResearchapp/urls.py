from django.urls import path
from . import views

urlpatterns = [
    path("",views.home, name='home'),
    path('summarize/', views.handle_pdf_upload, name='upload_pdf')
]