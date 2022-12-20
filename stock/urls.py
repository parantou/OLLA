from django.urls import path
from stock import views

app_name = 'stock'


urlpatterns = [
    path("join", views.joinFunc), 
    path("login", views.loginFunc), 
    path("logout", views.logoutFunc), 
    
    path('list', views.listFunc),
    path('insert', views.insertFunc), 
    path('insertok', views.insertOkFunc),
    path('search', views.searchFunc),
    path('content', views.contentFunc),
    path('update', views.updateFunc), 
    path('updateok', views.updateOkFunc),
    path('delete', views.deleteFunc),
    path('deleteok', views.deleteOkFunc),
    
    path('commentinsert', views.commentInsert),     
    path('commentdelete', views.commentDelete),
   
]
 