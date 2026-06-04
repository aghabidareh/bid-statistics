from django.urls import path

from statistical_tables import views

app_name = "statistical_tables"

urlpatterns = [
    path("", views.index, name="index"),
    path("<slug:slug>/", views.show, name="show"),
]
