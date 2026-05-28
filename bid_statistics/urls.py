from django.urls import include, path

urlpatterns = [
    path("", include("test_statistics.urls")),
]
