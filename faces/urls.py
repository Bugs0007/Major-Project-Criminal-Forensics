from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_face, name='upload_face'),
    path('search/', views.search_face, name='search_face'),
    path('list/', views.list_faces, name='list_faces'),
    path('delete/<int:face_id>/', views.delete_face, name='delete_face'),
    path('health/', views.health_check, name='health_check'),
]