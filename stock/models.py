from django.db import models

# Create your models here.

# 임시로 만든 멤버 테이블 필요없으면 삭제
class Member(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    pwd = models.CharField(max_length=50)
    name = models.CharField(max_length=50)
    phone = models.CharField(max_length=50)
    email = models.CharField(max_length=50)
    reg_date = models.DateField(auto_now_add=True)
    

class BoardTab(models.Model):
    bt_id = models.CharField(max_length=50)
    bt_pwd = models.CharField(max_length = 50)
    title = models.CharField(max_length = 100)
    content = models.TextField()
    reg_date = models.DateField(auto_now_add=True)
    readcnt = models.IntegerField()
    
    
    
class Comment(models.Model):
    user = models.ForeignKey('Member', on_delete=models.CASCADE)
    post = models.ForeignKey('BoardTab', on_delete=models.CASCADE)
    content = models.TextField()
    reg_date = models.DateField(auto_now_add=True)
