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
    # data = pd.read_csv('/static/models/주가+감성 데이터.csv')
    stockName='기아'
    # 테스트해볼 10일데이터: 정규화를 미리해놓은 데이터를 쓰면 안되고 여기서 써야 나중에 inverse도 쓸수있다.
    x = [[  65300.   ,     812308.  ,           0.   ,          0.008762,
     2408.27002 ,     717.900024 ,  33849.460938 ,  11049.5,
     3963.939941  ,  1334.560059 ,      9.574941],
    [  65500. ,      1411202. ,            0.029052  ,    -0.000828,
     2433.389893 ,    727.539978 ,  33852.53125 ,   10983.780273,
     3957.629883 ,   1340.119995  ,     9.65901 ],
    [  66800. ,      1970987.  ,           0.019316  ,     0.005606,
     2472.530029,     729.539978,   34589.769531 ,  11468.,
     4080.110107 ,   1325.369995,       9.551941],
 [  69000.    ,   1086698.   ,          0.   ,          0.009106,
     2479.840088 ,    740.599976,   34395.011719 ,  11482.450195,
     4076.570068  ,  1279.800049  ,     9.312411],
 [  68700.  ,     1451053.  ,          -0.030612  ,    -0.000394,
     2434.330078  ,   732.950012 ,  34429.878906 ,  11461.5,
     4071.699951 ,   1303.25   ,        9.642597],
 [  66800.  ,     1148365.  ,          -0.013534  ,     0.003971,
     2419.320068 ,    733.320007 ,  33947.101563,   11239.94043,
     3998.840088  ,  1299.170044  ,     9.660693],
 [  65200.  ,      825203. ,           -0.007622 ,      0.003429,
     2393.159912 ,    719.440002 ,  33596.339844 ,  11014.889648,
     3941.26001  ,   1304.369995 ,      9.543659],
 [  65000. ,       743679.  ,          -0.001536 ,      0.001391,
     2382.810059,     718.140015,   33597.921875 ,  10958.549805,
     3933.919922  ,  1319.709961 ,      9.641189],
 [  65200.    ,   1271219.  ,          -0.004615  ,    -0.001748,
     2371.080078  ,   712.52002  ,  33781.480469,   11082.,
     3963.51001  ,   1314.099976  ,     9.635365],
 [  65100. ,      1011844.   ,          0.007728  ,     0.003641,
     2389.040039  ,   719.48999 ,   33781.480469  , 11082.,
     3963.51001  ,   1316.630005  ,     9.637627]]
    y =[65200]
    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x)
    scaled_y = scaler.fit_transform(np.array(y).reshape(1,-1))
    print(scaled_x)
    print(scaled_y)
    # test_x = [[[0.7533039647577096, 0.1611144407042807, 0.5660338736224322, 0.31171768264244265, 0.4213458792502074, 0.47218813042402275, 0.5711026644849282, 0.442987034962536, 0.5167892347841567, 0.45740580996967406, 0.32784351496479136], [0.7268722466960353, 0.23860102381424045, 0.7347177477410248, 0.4452055891820311, 0.42610506791575675, 0.4708282123719916, 0.6236629341607127, 0.4911467699323917, 0.5765498640488582, 0.4947308234473864, 0.37597375644177333], [0.8105726872246697, 0.06463541626025635, 0.5106608534345883, 0.4582649291016724, 0.4324890127515295, 0.47896123201751584, 0.6720843143081456, 0.5011216655286779, 0.5970908039058291, 0.5181283926919029, 0.4020338903780436], [0.748898678414097, 0.41169822424587654, 0.1890694915125477, 0.30504478911191774, 0.41298273474664393, 0.45981845134932176, 0.6508199525013056, 0.4714940013664659, 0.5715398496583148, 0.5100957663573515, 0.3296236805874253], [0.5814977973568283, 0.4038412793424069, 0.32384104994976787, 0.2914059450053857, 0.40334983930385393, 0.4562619117767457, 0.653138311246483, 0.47639857145932574, 0.5794938932336087, 0.5242076211213957, 0.314702735725791], [0.5726872246696035, 0.26214844928917735, 0.614011939620793, 0.29769617914087443, 0.38577695509495413, 0.4251679593684301, 0.6169362104560099, 0.4295289850601147, 0.5341811476343881, 0.5687178730079276, 0.3364841875107789], [0.537444933920705, 0.10442230663531005, 0.5783153635389787, 0.33705744446146624, 0.3512378530608724, 0.37731108484432374, 0.53728326763125, 0.3712124105964527, 0.45998064106711656, 0.5980102601510886, 0.3138879226431808], [0.5594713656387666, 0.13279002091980172, 0.5547734202204269, 0.3093904197519128, 0.32016528581721104, 0.34475266665805915, 0.5182076067766603, 0.3711636692756204, 0.4523873438360515, 0.6293293576797243, 0.3316932327398474], [0.5682819383259914, 0.15834485126451406, 0.4608489927705599, 0.3036207524488642, 0.3340196949323446, 0.3701718282331541, 0.5255942305916346, 0.3802144158574343, 0.46226003225193235, 0.613559164070737, 0.3513057470995573], [0.5550660792951545, 0.12050797537549798, 0.8513526861856813, 0.38556643653628286, 0.36812418918899104, 0.4073850967522672, 0.5655428609165707, 0.41764578876943004, 0.5101064120577665, 0.6169491579256352, 0.33869331750963916]]]
    test_y = [[0.6486486486486487]]

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