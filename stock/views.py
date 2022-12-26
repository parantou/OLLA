from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime, timedelta, date
from stock.models import Member, Comment, BoardTab
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.contrib.auth.decorators import login_required
from sklearn.preprocessing import MinMaxScaler
from urllib import parse
from dateutil.relativedelta import relativedelta
from django.conf import settings
from bs4 import BeautifulSoup
from argparse import Namespace
from sklearn.metrics import classification_report
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import pandas as pd
import numpy as np
import os
import requests
import re
import FinanceDataReader as fdr
import torch
import torch.nn as nn

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
        # print(id + '/' + pwd)
        # print(rs)
 
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

'''
아나콘다에서 필수 설치 : 
1. conda install pytorch torchvision torchaudio cpuonly -c pytorch
2. pip install -U finance-datareader
3. pip install transformers
'''
def stockShow(request):
    #입력값 가져오기
    stockName = request.POST.get('stockName')
    stockName_en = parse.quote(stockName,encoding='EUC-KR')
    
    df_KOSPI = fdr.StockListing('KOSPI') 
    #입력한 종목명의 종목코드 찾기
    try:
        stockName=df_KOSPI.loc[df_KOSPI['Name'] == stockName]['Code'].values[0]
    except Exception as e:
        print('error : ',e)
        return render(request, 'show.html', {'NotFound':'존재하지 않는 종목입니다. 다시 검색해주세요.'})
    
    # 10일치 주가 데이터
    stock_dataset=getStockData(stockName)   
    
    #이전 행의 값으로 결측치 채우기
    stock_dataset.fillna(method= 'ffill',inplace=True)
    
    #뉴스==========================================================
    url = "https://finance.naver.com/news/news_search.naver?rcdate=&q="+stockName_en+"&x=0&y=0&sm=title.basic&pd=4"
    
    start_date=stock_dataset.head(1)['Date'][0].strftime("%Y-%m-%d")
    end_date=stock_dataset.tail(1)['Date'][9].strftime("%Y-%m-%d")

    financialNews(url +"&stDateStart="+start_date+"&stDateEnd="+end_date+"&page=")
    NaverFinanceSI_dataset = financialNews(url +"&stDateStart="+start_date+"&stDateEnd="+end_date+"&page=")
    
    NaverFinanceSI_dataset.reset_index(inplace=True)
    NaverFinanceSI_dataset['Date'] = pd.to_datetime(NaverFinanceSI_dataset['Date'])
    
    NaverFinanceF = pd.merge(stock_dataset,NaverFinanceSI_dataset,how='left',left_on='Date', right_on='Date').drop(columns=['negative','positive','logit'])
    NaverFinanceF = NaverFinanceF.fillna(0) # 10일데이터 데이터셋 : NaverFinanceF
    
    NaverFinanceF = NaverFinanceF.drop(columns='intensity')
    
    #예측==========================================================
    NaverFinanceF.set_index('Date',append=True,inplace=True)
    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(NaverFinanceF.drop(columns=['Close','High','Low']))
    scaled_y = scaler.fit_transform(NaverFinanceF[['Close']])
    
    #list type로 변경
    x = scaled_x.tolist() 
    y = scaled_y.tolist()
    models_folder = settings.BASE_DIR / 'stock' / 'static' / 'dlmodel' #피클된 모델 파일이 들어있는 폴더를 찾을수있도록 BASE_DIR로 경로 설정
    file_path = os.path.join(models_folder, os.path.basename('kia_model.pickle')) #입력받은 경로의 기본 파일 이름 반환받음
    model = pickle.load(open(file_path, "rb"))
    
    pred_y = model.predict(np.array(x).reshape(-1,10,11))
    result=int(scaler.inverse_transform(pred_y.reshape(1,-1))[0][0])
    
    print('result : ',result) #62582
    if result >= 1000 and result < 5000 :
        pass
    elif result >= 5000 and result < 10000 :
        pass
    elif result >= 10000 and result < 50000 :
        pass
    elif result >= 50000 and result < 100000 :
        pass
    elif result >= 100000 and result < 500000 :
        pass
    elif result >= 500000:
        pass
        
    
    url= cloudShow(file_path)
    
    #stock_close_df, pred은 dataframe type
    stock_close_df, pred = graphShow(file_path, stockName, result)
    result_date = np.array(stock_close_df['Date']).tolist() #날짜
    pred_Close = np.array(stock_close_df['Close']).tolist() #종가+예측
    real_Close = np.array(stock_close_df['Close'][:-1]).tolist() #종가
    
    return render(request, 'show.html', {'result':result,'url':url, 'result_date': result_date, 'pred_Close': pred_Close,  'real_Close':real_Close})

# 10일치 주가 및 보조 데이터 추출
def getStockData(stockName):
    start_date= datetime.today() - timedelta(20)
    end_date= datetime.today()
    df = fdr.DataReader(stockName, start_date.strftime("%Y-%m-%d"),end_date.strftime("%Y-%m-%d"))
    df_ten = df.tail(10)
    df_ten.reset_index(inplace=True)
    #FinanceDataReader를 이용해 보조 데이터 가져오기
    df_dict = {
        'KS11' : fdr.DataReader('KS11', start_date, end_date), #코스피지수
        'KQ11' : fdr.DataReader('KQ11',start_date, end_date), #코스닥지수
        'DAU' : fdr.DataReader('DJI', start_date, end_date), #다우지수
        'NASDAQ' : fdr.DataReader('IXIC', start_date, end_date), #나스닥지수
        'SP500':  fdr.DataReader('US500', start_date, end_date), #S&P 500
        'USD' : fdr.DataReader('USD/KRW', start_date, end_date), #달러당 원화 환율
        'JPY' : fdr.DataReader('JPY/KRW', start_date, end_date) #엔화 원화 환율
    }

    #각 보조데이터의 수정주가를 추출해서 stock_data에 추가하기
    for key in df_dict:
      #df_dict[key] 인덱스 해제
      extra_df = df_dict[key].rename_axis('Date').reset_index()
      
      #사용할 칼럼명 생성
      newName= key+'_Adj_Close'

      #Date와 Adj Close컬럼만 추출
      extra_df_cut=extra_df.loc[:,['Date','Adj Close']]
      
      #Adj Close컬럼명을 newName으로 변경
      extra_df_cut.rename(columns={'Adj Close':newName},inplace=True)

      #stock_data의 Date를 기준으로 extra_df_cut과 merge
      merge_outer = pd.merge(df_ten,extra_df_cut, how='left',left_on='Date', right_on='Date')
      
      #stock_data에 merge_outer의 newName(key+'_Adj_Close')칼럼 데이터 추가
      df_ten[newName] = merge_outer[newName]
    return df_ten

def financialNews(url):
    #컬럼스 : 종목명, 뉴스 제목 , 언론사, 작성날짜, 시간, 본문 내용
    stocklist=[]
    titlelist=[]
    companylist=[]
    datelist=[]
    totalsa_list=[] # 뉴스(제목) sa(sentiment analysis)
    # timelist=[]
    # contentlist=[]
    negative_list=[]
    positive_list=[]

    #header에 User Agent 정보(네이버의 경우 네이버 검색로봇 Yeti)를 기입하여 웹 서버로 페이지를 요청할 시에 같이 보내서 크롤링 제한을 피한다
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Yeti/1.1; +http://naver.me/spd)'}

    # 감성수치 사전학습 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
    model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")
    sa = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) # 감성분류 모델 객체 sa 생성

    #i는 페이지 번호 | 1~200 반복
    for i in range(1,201) :
        #본문으로 들어가는 링크를 담을 리스트
        # links=[] 
        res = requests.get(url+str(i), headers = headers)
        soup = BeautifulSoup(res.text, 'html.parser') 
        body = soup.find(class_='newsSchResult')
        title = body.find_all(['dt','dd'], {'class': 'articleSubject'})  #뉴스제목
        
        #다음 페이지로 넘어갔을 때, 읽어올 title이 없을 경우 for문 종료
        if title == [] : break

        summary = body.find_all('dd', {'class': 'articleSummary'}) #요약문,언론사,날짜,시간
        stock = soup.select('p.resultCount > strong')[0].text  #종목명

        #한 페이지의 20개의 제목 가져오기
        for item in title:
          information = item.text 
          
          #link 추출 및 리스트에 저장
          # href = item.find('a').attrs['href']
          # links.append('https://finance.naver.com'+href)

          #제목 리스트에 저장
          titlelist.append(information.strip())
          #제목 가져올때마다 종목명 리스트안에 추가
          stocklist.append(stock) 
          
          # print('제목 : ',information)

        #한 페이지의 20개의 언론사 , 작성날짜, 시간 가져오기
        for item in summary:
          # information = item.string # 요약본, 언론사, 날짜, 시간
          press =  re.sub(r"\s", "", item.find('span', {'class': 'press'}).text) #언론사 -> 공백 제거완료
          wdate_all =  re.sub(r"\s", "", item.find('span', {'class': 'wdate'}).text) #날짜,시간 -> 공백 제거완료
          wdate=wdate_all[:10] #날짜만
          # time=wdate_all[10:] #시간만

          #언론사 리스트에 저장
          companylist.append(press)
          #작성날짜 리스트에 저장
          datelist.append(wdate) 
          #작성시간 리스트에 저장
          # timelist.append(time)

        #본문 추출
        '''
        for link in links:
          print('link : ',link)
          res = requests.get(link, headers = headers)
          soup = BeautifulSoup(res.text, 'html.parser')
          try:
              # link_news 내용이 있을 경우
              if soup.find('div',{'class': 'link_news'}) != None:
                  # link_news 제거
                  soup.find('div',{'class': 'link_news'}).decompose()
              #본문 내용 추출
              content = soup.find('div',{'id':'content'}).text.strip()
          except AttributeError as e:
              #본문이 없을 경우, NaN로 채우기
              print('error : ',e)

          print(content)
          print("-------다음뉴스------")
          #본문내용을 리스트에 저장
          contentlist.append(content)
        '''  

        #totalsa_list = [] # 뉴스(제목+내용) sa(sentiment analysis) 배열 초기화
    for news in titlelist:
        inputs = tokenizer(news, return_tensors='pt')
        output = model(**inputs)
        output = output.logits.tolist()[0]
        totalsa_list.append(output)
           
    outputs = torch.tensor(totalsa_list) # 출력값을 텐서로 변환 # outputs
    predictions = nn.functional.softmax(outputs, dim=-1) # 출력값을 확률 형태로 변환 # predictions
    sa = predictions.detach().numpy()
    # detach()는 tensor에서 이루어진 모든 연산을 분리한 tensor을 반환하는 method
    #print(predictions.detach().numpy())
    negative_list = sa[:, 0]
    positive_list = sa[:, 2]
    #print(negative_list)
    
    # 데이터셋 만들기
    NaverFinance = pd.DataFrame({
        'stock' : stocklist,
        'title' : titlelist,
        'company' : companylist,
        'Date' : datelist,
        # 'time' : timelist,
        # 'content' : contentlist
        'negative' : negative_list,
        'positive' : positive_list
    })

    NaverFinanceSI = NaverFinance.groupby(['Date']).mean()
    NaverFinanceSI['logit'] = np.log(NaverFinanceSI['positive']/NaverFinanceSI['negative']) # logit = ln(P/N)
    avgn = NaverFinance['title'].count()/NaverFinanceSI['logit'].count() # 전체 기간에 작성된 뉴스 수의 평균
    NaverFinanceSI['intensity'] = np.log(1 + (NaverFinance.groupby(['Date'])['title'].count()/avgn))
    NaverFinanceSI['sent_index'] = NaverFinanceSI['logit'] * NaverFinanceSI['intensity']

    # #결과 출력
    # print(NaverFinanceSI)
    # # csv 파일로 저장      
    # #NaverFinance.to_csv('NewFinance10.csv', mode='w', encoding='utf-8-sig') 
    return NaverFinanceSI

def cloudShow(file_path):
    if "kia_model" in str(file_path):
        url = "KIA_wcloud"
    return url

def graphShow(file_path, stockName, result):
    if "kia_model" in str(file_path):
        
        start_date = datetime.today()  # 오늘 날짜
        print(start_date)
        print(start_date.weekday())
        d_day = 100
        
        target_date = (start_date - timedelta(d_day)).strftime("%Y-%m-%d")  # 100일전 날짜
        # 입력받은 주가로 100일전 ~ 오늘 날 까지의 주가 데이터 출력
        df =fdr.DataReader(stockName, target_date, start_date.strftime("%Y-%m-%d"))
        pred = result
        stock_close_df = df[['Close']]
        stock_close_df.reset_index(inplace=True)

        # 데이터 추가해서 원래 데이터프레임에 저장하기
        if start_date.weekday() > 4:
            start_date += date.timedelta(days=8 - start_date.isoweekday())
            print(start_date)
            dict_data = pd.DataFrame({'Date':[start_date],'Close':[pred]})
            stock_close_df = stock_close_df.append(dict_data)
        else:
            start_date += timedelta(days=1)
            print('start_date', start_date)
            dict_data = pd.DataFrame({'Date':[start_date],'Close':[pred]})
            stock_close_df = stock_close_df.append(dict_data)
            
            stock_close_df['Date'] = stock_close_df['Date'].dt.strftime("%Y-%m-%d")
        print('---------------------')
        print(stock_close_df) #전체
        print(stock_close_df[-1:]) #다음날
        print(stock_close_df[:-1]) # 백일
        pred = stock_close_df[-1:]

    return stock_close_df, pred