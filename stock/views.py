from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from stock.models import Member, Comment
from stock.models import BoardTab 
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.contrib.auth.decorators import login_required




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
