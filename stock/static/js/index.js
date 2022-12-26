const form=document.querySelector("#search_form");
$('.search-btn').click(function(){
	if($('.search-txt').val() == ''){
		alert('종목명을 입력해주세요.')
	}else{
		clearTxt($('.result_box'))
		extend()
		fetchFunc(form)
	}			
});
$('.search-txt').keydown(function(e){
	if (e.keyCode === 13){
		e.preventDefault();
		if($('.search-txt').val() == ''){
			alert('종목명을 입력해주세요.')
			return
		}else{
			clearTxt($('.result_box'))
			extend()
			fetchFunc(form)
		}
	}
});
function extend(){
	$('.index_txt').slideUp(1000)
	$('.search-txt').addClass('input_expend');
	$('#img_box').addClass('txt_fadeout')
	setTimeout(function(){
		$('#img_box').css({'display':'none'})
		$('.result_box').slideDown()
	}, 1000)
	setTimeout(function(){
		$('.result_box').slideDown()
	}, 800)
}

function fetchFunc(form){
	openLoading()
	ajaxFormPromise(form)
    .then(function(res){
		return res.text();
	}).then(function(data){
		closeLoading()
		$('.result_box').html(data);
		//document.querySelector('#addPwdForm').innerHTML+=data;
	});
}

function clearTxt(el){
	el.fadeOut()
	setTimeout(function(){
		el.text('')
	}, 500)
}

//함수의 인자로 ajax 전송할 폼의 참조값을 넣어주면 알아서 ajax 전송되도록 하는 함수 
function ajaxFormPromise(form){
	const url=form.getAttribute("action");
	const method=form.getAttribute("method");
	
	// Promise 객체를 담을 변수 만들기 
	let promise;
	//파일 업로드 폼인지 확인해서
	if(form.getAttribute("enctype") == "multipart/form-data"){
		//폼에 입력한 데이터를 FormData 에 담고 
		let data=new FormData(form);
		// fetch() 함수가 리턴하는 Promise 객체를 
		promise=fetch(url,{
			method:"post",
			body:data
		});
		return promise;//리턴해 준다 (여기서 함수가 종료 된다.)			
	}
	
	const queryString=new URLSearchParams(new FormData(form)).toString();
	
	if(method=="GET" || method=="get"){//만일 GET 방식 전송이면 
		//fetch() 함수를 호출하고 리턴되는 Promise 객체를 변수에 담는다. 
		promise=fetch(url+"?"+queryString);
	}else if(method=="POST" || method=="post"){//만일 POST 방식 전송이면
		//fetch() 함수를 호출하고 리턴되는 Promise 객체를 변수에 담는다. 
		promise=fetch(url,{
			method:"POST",
			headers:{"Content-Type":"application/x-www-form-urlencoded; charset=utf-8"},
			body:queryString
		});
	}
	return promise;
}

// 로딩창 키는 함수
function openLoading() {
    //화면 높이와 너비를 구합니다.
    let maskHeight = $(document).height();
    let maskWidth = window.document.body.clientWidth;
    //출력할 마스크를 설정해준다.
    let mask ="<div id='mask' style='position:absolute; z-index:9000; display:none; left:0; top:0;'></div>";
    // 로딩 이미지 주소 및 옵션
    let loadingImg ='';
    loadingImg += "<div id='loadingImg' style='position:absolute; top: calc(50% - (200px / 2)); width:100%; z-index:99999999;'>";
    loadingImg += " <img src='/static/images/loading.gif' style='position: relative; display: block; margin: 0px auto;'/>";
    /* https://loadingapng.com/animation.php?image=4&fore_color=000000&back_color=FFFFFF&size=30x30&transparency=1&image_type=0&uncacher=75.5975991029623 */
    loadingImg += "</div>"; 
    //레이어 추가
    $('body')
        .append(mask)
        .append(loadingImg)
    //마스크의 높이와 너비로 전체 화면을 채운다.
    $('#mask').css({
            'width' : maskWidth,
            'height': maskHeight,
            'opacity' :'0.3'
    });
    //마스크 표시
    $('#mask').show();  
    //로딩 이미지 표시
    $('#loadingImg').show();
}

// 로딩창 끄는 함수
function closeLoading() {
    $('#mask, #loadingImg').hide();
    $('#mask, #loadingImg').empty(); 
}