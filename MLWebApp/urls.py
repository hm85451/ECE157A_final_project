from MLWebApp import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'MLWebApp'
urlpatterns = [
    path('', views.MLWebAppView.as_view(), name='MLWebApp'),
    path('<datasetid>/', views.delete, name='delete'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)