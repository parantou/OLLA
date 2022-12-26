from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from stock.models import Member, Comment, BoardTab
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.contrib.auth.decorators import login_required
import pickle
import pandas as pd
import numpy as np
from django.conf import settings
import os
from sklearn.preprocessing import MinMaxScaler
from _ast import If
# Create your views here.
def mainFunc(request):
    return render(request,'main.html')

#user

def joinFunc(request):
    if request.method == "GET":
        return render(request, 'user/join.html')
    elif request.method == "POST":
        context = {}
        id = request.POST["id"]
        pwd = request.POST["pwd"]
        name = request.POST["name"]
        phone = request.POST["phone"]
        email = request.POST["email"]
 
        # 회원가입 중복체크
        rs = Member.objects.filter(id=id)
        if rs.exists():
            context['message'] = id + "가 중복됩니다."
            return render(request, 'user/join.html', context)
 
        else:
            Member.objects.create(
                id=id, pwd=pwd,  name=name, phone=phone, email=email,
                reg_date=datetime.now())
            context['message'] = name + "님 회원가입 되었습니다."
            return render(request, 'main.html', context)

def loginFunc(request):
    if request.method == "GET":
        return render(request, 'user/login.html')
    elif request.method == "POST":
        context = {}
        
        id = request.POST.get('id')
        pwd = request.POST.get('pwd')
 
        # 로그인 체크하기
        rs = Member.objects.filter(id=id, pwd=pwd).first()
        print(id + '/' + pwd)
        print(rs)
 
        #if rs.exists():
        if rs is not None:
 
            # OK - 로그인
            request.session['m_id'] = id
            request.session['m_name'] = rs.name
            
 
            context['m_id'] = id
            context['m_name'] = rs.name
            context['message'] = rs.name + "님이 로그인하셨습니다."
            return render(request, 'main.html', context)

        else:
 
            context['message'] = "로그인 정보가 맞지않습니다.\\n\\n확인하신 후 다시 시도해 주십시오."
            return render(request, 'user/login.html', context)

def logoutFunc(request):
    request.session.flush()
    return redirect('/')

def profileFunc(request):
    profile_data = Member.objects.get(id=request.session.get('m_id'))
    return render(request, 'user/profile.html', {'profile_data':profile_data})

def upProfileFunc(request):
    profile_data = Member.objects.get(id=request.session.get('m_id'))
    return render(request, 'user/upProfile.html', {'profile_data':profile_data}) 

def upProfileOkFunc(request):
    try:
        profile_data = Member.objects.get(id=request.session.get('m_id'))
        upProfile = Member.objects.get(id=request.session.get('m_id'))
        # 비밀번호 비교 후 수정 여부 결정
        print(request.POST.get('pwd'))
        if upProfile.pwd == request.POST.get('pwd'):
            upProfile.name = request.POST.get('name')
            upProfile.phone = request.POST.get('phone')
            upProfile.email = request.POST.get('email')
            upProfile.save()
        else:
            return render(request, 'user/upProfile.html', {'message':'비밀번호가 일치하지 않습니다.', 'profile_data':profile_data})
    except Exception as e:
        return render(request, 'board/error.html')
    
    return redirect('/user/profile')   # 수정 후 목록 보기

def delProfileFunc(request):
    try:
        del_profile = Member.objects.get(id=request.session.get('m_id'))
    except Exception as e:
        return render(request, 'board/error.html')
    
    return render(request, 'user/delProfile.html', {'del_profile':del_profile})   
    
def delProfileOkFunc(request):
    del_profile = Member.objects.get(id=request.session.get('m_id'))
    
    if del_profile.pwd == request.POST.get('del_pwd'):
        del_profile.delete();
        return redirect('/user/logout')   # 삭제 후 목록 보기
    else:
        return render(request, 'user/delProfile.html', {'message':'비밀번호가 일치하지 않습니다.', 'del_profile':del_profile})


#board

def listFunc(request):
    data_all = BoardTab.objects.all().order_by('-id')  
    
    paginator = Paginator(data_all, 10)
    page = request.GET.get('page')
    
    try:
        datas = paginator.page(page)
    except PageNotAnInteger:
        datas = paginator.page(1)
    except EmptyPage:
        datas = paginator.page(paginator.num_pages)
    
    return render(request, 'board/board.html', {'datas':datas})


def insertFunc(request):
    if not 'm_id' in request.session:
        return render(request, 'user/login.html')
    else:
        return render(request, 'board/insert.html')


def insertOkFunc(request):
    if request.method == 'POST':
        try:
            datas = BoardTab.objects.all()
            BoardTab(
                bt_id = request.POST.get('bt_id'),
                bt_pwd = request.POST.get('bt_pwd'),
                title = request.POST.get('title'),
                content = request.POST.get('content'),
                reg_date = datetime.now(),
                readcnt = 0,

            ).save()
        except Exception as e:
            print('insert err : ', e)
            return render(request, 'board/error.html')
    
    return redirect('/board/list')   # 추가 후 목록 보기


def searchFunc(request):
    if request.method == 'POST':
        s_type = request.POST.get('s_type')
        s_value = request.POST.get('s_value')
        # print(s_type, s_value)
        # SQL의 like 연산 --> ORM에서는 __contentains=값
        if s_type == 'title':
            datas_search = BoardTab.objects.filter(title__contains=s_value).order_by('-id')
        elif s_type == 'bt_id':
            datas_search = BoardTab.objects.filter(bt_id__contains=s_value).order_by('-id')
        
        paginator = Paginator(datas_search, 5)
        page = request.GET.get('page')
        
        try:
            datas = paginator.page(page)
        except PageNotAnInteger:
            datas = paginator.page(1)
        except EmptyPage:
            datas = paginator.page(paginator.num_pages)
        
        return render(request, 'board/board.html', {'datas':datas})


def contentFunc(request):
    page = request.GET.get('page')
    data = BoardTab.objects.get(id=request.GET.get('id'))
    
    comment = Comment.objects.filter(post_id=request.GET.get('id')).order_by('-id') 
    data.readcnt = data.readcnt + 1   # 조회수 증가
    data.save()   # 조회수 update
    return render(request, 'board/content.html', {'data_one':data, 'page':page, 'comment_one':comment})


def updateFunc(request):   # 수정 화면
    try:
        data = BoardTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        return render(request, 'board/error.html')
    
    return render(request, 'board/update.html', {'data_one':data})


def updateOkFunc(request):  # 수정 처리
    try:
        upRec = BoardTab.objects.get(id=request.POST.get('id'))
        
        # 비밀번호 비교 후 수정 여부 결정
        if upRec.bt_pwd == request.POST.get('up_pwd'):
            upRec.bt_id = request.POST.get('bt_id')
            upRec.title = request.POST.get('title')
            upRec.content = request.POST.get('content')
            upRec.save()
        else:
            return render(request, 'board/update.html', {'data_one':upRec, 'msg':'비밀번호가 일치하지 않습니다.'})

    except Exception as e:
        return render(request, 'board/error.html')
    
    return redirect('/board/list')   # 수정 후 목록 보기


def deleteFunc(request):
    try:
        del_data = BoardTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        return render(request, 'board/error.html')
    
    return render(request, 'board/delete.html', {'data_one':del_data})


def deleteOkFunc(request):
    del_data = BoardTab.objects.get(id=request.POST.get('id'))
    
    if del_data.bt_pwd == request.POST.get('del_pwd'):
        del_data.delete();
        return redirect('/board/list')   # 삭제 후 목록 보기
    else:
        return render(request, 'board/error.html')

#comment
def commentInsert(request): 
    if not 'm_id' in request.session:
        return render(request, 'user/login.html')
    else:
        if request.method == 'POST':
            try:
                page = request.GET.get('page')
                data = BoardTab.objects.get(id=request.GET.get('id'))
                comment = Comment.objects.filter(post_id=request.GET.get('id')).order_by('-id')
                user = request.POST.get('user')
                id = request.GET.get('id')
                Comment(
                    user = Member.objects.get(id=user),
                    post = BoardTab.objects.get(id=id),
                    content = request.POST.get('content'),
                    reg_date = datetime.now(),
                ).save()
            except Exception as e:
                print('insert err : ', e)
                return render(request, 'board/error.html')
        
        return render(request, 'board/content.html', {'data_one':data, 'page':page,'comment_one':comment})   # 추가 후 게시글 보기

def commentDelete(request):
    try:
        page = request.GET.get('page')
        data = BoardTab.objects.get(id=request.GET.get('id'))
        comment = Comment.objects.filter(post_id=request.GET.get('id')).order_by('-id') 
        del_comment = Comment.objects.get(id=request.GET.get('comment_id'))
        del_comment.delete();    
        
        return render(request, 'board/content.html', {'data_one':data, 'page':page,'comment_one':comment})
    except Exception as e:
        return render(request, 'board/error.html')
    
def stockShow(request):

    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x)
    scaled_y = scaler.fit_transform(np.array(y).reshape(1,-1))
    print(scaled_x)
    print(scaled_y)

    models_folder = settings.BASE_DIR / 'stock' / 'static' / 'dlmodel' #피클된 모델 파일이 들어있는 폴더를 찾을수있도록 BASE_DIR로 경로 설정
    file_path = os.path.join(models_folder, os.path.basename('kia_model.pickle')) #입력받은 경로의 기본 파일 이름 반환받음
    model = pickle.load(open(file_path, "rb"))
    # print(file_path)
    # 새로운 데이터로 주가예측
    pred = model.predict(np.array(scaled_x).reshape(-1,10,11))
    
    # 스케일러 정규화 원복
    # scaler.inverse_transform(np.array(pred[-1]).reshape(1,-1))[0][0]
    print('예측값: ', pred)
    print('실제값: ', test_y)
    print(np.array(pred[0]))
    print('원복 예측값: ', scaler.inverse_transform(np.array(pred[0]).reshape(1,-1))[0][0])
    print('원복 실제값: ', scaler.inverse_transform(np.array(test_y[0]).reshape(1,-1))[0][0])
    

    result= int(scaler.inverse_transform(np.array(pred[0]).reshape(1,-1))[0][0])
    url= cloudShow(file_path)
    return render(request, 'show.html', {'result':result,'url':url})

def cloudShow(file_path):
    if "kia_model" in str(file_path):
        url = "KIA_wcloud"
    return url

# def graphShow(file_path,stockName):
#     if "kia_model" in str(file_path):
#
#     return url