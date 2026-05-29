from django.urls import path

from regression import views

app_name = "regression"

urlpatterns = [
    path("", views.index, name="index"),
    path("<slug:slug>/", views.show, name="show"),
    path("<slug:slug>/calculate/", views.calculate, name="calculate"),
]
