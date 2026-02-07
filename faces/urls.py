from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_face, name='upload_face'),
    path('search/', views.search_face, name='search_face'),
    path('list/', views.list_faces, name='list_faces'),
    path('delete/<int:face_id>/', views.delete_face, name='delete_face'),
    path('health/', views.health_check, name='health_check'),
    path('sketch/image-to-sketch/', views.image_to_sketch, name='image_to_sketch'),
    path('sketch/compose-face/', views.compose_face_from_features, name='compose_face'),
    path('sketch/feature-library/', views.get_feature_library, name='feature_library'),
]