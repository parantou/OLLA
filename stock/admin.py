from django.contrib import admin
from .models import Member
 
 
class MemberAdmin(admin.ModelAdmin):
    # 내가 출력하고 싶은 제목명인 필드명 member_id과 name 을 표시한다.
    list_display = ('id', 'name', 'phone', 'email', 'reg_date')
 
 
admin.site.register(Member, MemberAdmin)