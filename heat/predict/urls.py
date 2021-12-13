from . import views
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

app_name = "predict"

urlpatterns = [
    path('version/', views.home, name='home'),
	path('food/', views.food, name='food')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)