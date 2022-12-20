//유효성 여부를 저장할 변수를 만들고 초기값 대입 
	let isNameValid=false;
	let isIdValid=false;
	let isPwdValid=false;
	let isPhoneValid=false;
	let isEmailValid=false;
	
	
	document.querySelector("#name").addEventListener("input", function(){
		this.classList.remove("is-valid");
		this.classList.remove("is-invalid");
		//입력한 이름
		const inputName=this.value;
		//이름을 검증할 정규 표현식
		//한글만 가능
		const reg=/^[가-힣]+$/;
		if(!reg.test(inputName)){
			this.classList.add("is-invalid");
			isNameValid=false;
		}else{
			this.classList.add("is-valid");
			isNameValid=true;
		}
	});
	
	// id 를 입력 할때 마다 호출되는 함수 등록 
	document.querySelector("#id").addEventListener("input", function(){
		//input 요소의 참조값을 self 에 미리 담아 놓기 
		const self=this;
		//일단 2개의 클래스를 모두 제거 한다음 
		self.classList.remove("is-valid"); //여기서는 self말고 this 써도 상관 없음
		self.classList.remove("is-invalid");
		
		//1. 현재 입력한 아이디를 읽어와서
		const inputId=this.value;
		
		//아이디를 검증할 정규표현식 객체
		const reg=/^[a-z].{4,9}$/;
		//만일 입력한 아이디가 정규표현식을 통과 하지 못한다면 빨간색으로 표시하고 함수를 여기서 바로 종료 시키기 
		if(!reg.test(inputId)){
			self.classList.add("is-invalid"); //invalid-feedback이 뜬다.
			isIdValid=false;
			return;
		}
		
		
		
	});
	
	//비밀번호를 확인 하는 함수 
	function checkPwd(){
		document.querySelector("#pwd").classList.remove("is-valid");
		document.querySelector("#pwd").classList.remove("is-invalid");
		document.querySelector("#pwd2").classList.remove("is-valid");
		document.querySelector("#pwd2").classList.remove("is-invalid");
		
		const pwd=document.querySelector("#pwd").value;
		const pwd2=document.querySelector("#pwd2").value;
		//비밀번호를 검증할 정규 표현식
		//let reg=/[\W]/; 특수문자 하나는 꼭 들어가야 한다
		//최소 하나의 문자,숫자,특수문자가 들어가고 8글자~13글자 이내로 입력
		const reg=/^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&^])[A-Za-z\d@$!%*#?&^]{8,13}$/;
		//만일 비밀번호가 정규 표현식을 통과 하지 못한다면 
		if(!reg.test(pwd)){		
			document.querySelector("#pwd").classList.add("is-invalid");
			isPwdValid=false;
			return; //함수를 여기서 끝내라 
		}else{
			document.querySelector("#pwd").classList.add("is-valid");
		}
		
		if(pwd != pwd2){//만일 비밀번호 입력란과 확인란이 다르다면
			document.querySelector("#pwd2").classList.add("is-invalid");
			isPwdValid=false;
		}else{
			document.querySelector("#pwd2").classList.add("is-valid");
			isPwdValid=true;
		}
	}
	
	document.querySelector("#pwd").addEventListener("input", function(){
		//비밀번호를 검증하는 함수 호출
		checkPwd();
	});
	document.querySelector("#pwd2").addEventListener("input", function(){
		checkPwd();
	});
	
	document.querySelector("#email").addEventListener("input", function(){
		
		this.classList.remove("is-valid");
		this.classList.remove("is-invalid");
		//입력한 이메일
		const inputEmail=this.value;
		//이메일을 검증할 정규 표현식
		const reg=/@/;
		if(!reg.test(inputEmail)){
			this.classList.add("is-invalid");
			isEmailValid=false;
		}else{
			this.classList.add("is-valid");
			isEmailValid=true;
		}
	});
	
	document.querySelector("#phone").addEventListener("input", function(){
		
		this.classList.remove("is-valid");
		this.classList.remove("is-invalid");
		//입력한 전화번호
		this.value=this.value.replace(/[^0-9]/g, '');
		const inputPhone = this.value;
		//전화번호 검증할 정규 표현식
		const reg=/^01([0|1|6|7|8|9])([0-9]{3,4})([0-9]{4})/;
		if (!reg.test(inputPhone)) {
			this.classList.add("is-invalid");
			isPhoneValid=false;
		}else{
			this.classList.add("is-valid");
			isPhoneValid=true;
		}
	});
	
	
	//폼에 submit 이벤트가 일어 났을때 실행할 함수 등록
	document.querySelector("#signupForm").addEventListener("submit", function(event){
		
		//폼 전체의 유효성 여부
		//and 연상자 이용
		let isFormValid = isNameValid && isIdValid && isPwdValid && isEmailValid && isPhoneValid && isAddrValid01 && isAddrValid02;
		if(!isFormValid){
			//폼 제출 막기 
			//기본 동작을 막는 함수
			event.preventDefault();
			if(!isNameValid){
				alert('이름을 입력해주세요');
			}else if(!isEmailValid){
				alert('이메일을 입력해주세요');
			}else if(!isIdValid){
				alert('아이디를 입력해주세요');
			}else if(!isPwdValid){
				alert('비밀번호를 입력해주세요');
			}else if(!isAddrValid01){
				alert('주소를 선택해주세요');
			}else if(!isAddrValid02){
				alert('상세주소를 입력해주세요');
			}else if(!isPhoneValid){
				alert('핸드폰 번호를 입력해주세요');
			}
		}
		let postcode=document.querySelector("#postcode").value;
		let addr=document.querySelector("#addr").value;
		let detailAddr=document.querySelector("#detailAddr").value;
		let extraAddr=document.querySelector("#extraAddr").value;
		let totalAddr=document.querySelector("#totalAddr");
		totalAddr.value=postcode+"_"+addr+"_"+detailAddr+"_"+extraAddr;
	});