from django.urls import path

from test_statistics import views

app_name = "test_statistics"

urlpatterns = [
    path("", views.index, name="index"),
    path("<slug:slug>/", views.show, name="show"),
    path("<slug:slug>/calculate/", views.calculate, name="calculate"),
]
