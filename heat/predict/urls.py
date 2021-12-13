from django.urls import path
from . import views

app_name = "predict"

urlpatterns = [
    path('version/', views.home, name='home'),
	path('food/', views.food, name='food')
]