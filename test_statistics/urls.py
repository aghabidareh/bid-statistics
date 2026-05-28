from django.urls import path

from test_statistics import views

app_name = "test_statistics"

urlpatterns = [
    path("", views.home, name="home"),
    path("test-statistics/<slug:slug>/", views.show, name="show"),
    path("test-statistics/<slug:slug>/calculate/", views.calculate, name="calculate"),
]
