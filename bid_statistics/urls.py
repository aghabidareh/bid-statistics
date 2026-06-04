from django.urls import include, path

from bid_statistics import views

urlpatterns = [
    path("", views.home, name="home"),
    path("test-statistics/", include("test_statistics.urls")),
    path("regression/", include("regression.urls")),
    path("statistical-tables/", include("statistical_tables.urls")),
]
