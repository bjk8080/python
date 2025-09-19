address_book = {} #비어있는 주소록을 전역으로 생성한다.

def add(name, phone, email, address): #연락처의 이름을 키로 가지는 연락처 생성 코드
    address_book[name] = { #address_book의 키는 name
        "phone": phone,
        "email": email,
        "address": address #키를 받고 내주는 값의 목록
    }
    print(f"[추가 완료] {name}님의 연락처가 추가되었습니다.")

def delete(name): #연락처의 이름을 키로 받아 그 연락처를 삭제하는 코드
    if name in address_book:
        del address_book[name] #del을 사용하여 키를 받아 삭제가 가능하다.
        print(f"[삭제 완료] {name}님의 연락처가 삭제되었습니다.")
    else:
        print("[오류] 해당 이름이 주소록에 없습니다.")

def search(name): #연락처의 이름을 받아 그 연락처의 정보를 받아 오는 코드
    if name in address_book:
        print(f"이름: {name}")
        for key, value in address_book[name].items(): #items를 사용하면 키 값 쌍으로 반환해준다.
            print(f"{key}: {value}")
    else:
        print("[오류] 해당 이름이 주소록에 없습니다.")

def update(name, phone=None, email=None, address=None): #연락처의 이름을 받아 충족하는 데이터의 값을 수정하는 코드
    if name in address_book:
        if phone:
            address_book[name]["phone"] = phone
        if email:
            address_book[name]["email"] = email
        if address:
            address_book[name]["address"] = address
        print(f"[수정 완료] {name}님의 연락처가 수정되었습니다.")
    else:
        print("[오류] 해당 이름이 주소록에 없습니다.")

def show_all(): #모든 연락처를 받아서 보는 코드
    if not address_book: #연락처에 아무것도 없을 때 받는 코드
        print("[알림] 주소록이 비어 있습니다.")
        return
    print("\n===== 주소록 목록 =====")
    for name, info in address_book.items(): # 이름을 받아 주소안의 키-값 쌍 구조를 모두 불러온다.
        print(f"이름: {name}") #키 값을 말함
        for key, value in info.items(): 
            print(f"{key}: {value}") # 키값에 해당하는 키-값 쌍을 모두 출력한다.
        print("------------------")

# 사용자 메뉴 폼을 제작
while True:
    print("\n===== 주소록 메뉴 =====")
    print("1. 연락처 추가")
    print("2. 연락처 삭제")
    print("3. 연락처 검색")
    print("4. 연락처 수정")
    print("5. 모든 연락처 보기")
    print("6. 모든 값을 삭제") #6번은 딕셔너리 조작을 통해 한번 응용해보았다.
    print("0. 종료")
    # 각 버튼을 눌러 사용자가 하고 싶은 능력을 실행한다.
    choice = input("선택: ")
    
    if choice == "1":
        name = input("이름: ")
        phone = input("전화번호: ")
        email = input("이메일: ")
        address = input("주소: ")
        add(name, phone, email, address) #맨처음 키를 받아 그 키 안에 값을 다 입력한 후 add 함수로 시행한다.
    elif choice == "2":
        name = input("삭제할 이름: ") #키값을 받아 그 키값을 delete 함수에서 키 값에 해당하는 목록을 제거
        delete(name)
    elif choice == "3":
        name = input("검색할 이름: ") #키값을 받아 그 키값을 search 함수에서 키 값에 해당하는 목록을 불러옴.
        search(name)
    elif choice == "4":
        name = input("수정할 이름: ")
        phone = input("새 전화번호 (없으면 Enter): ")
        email = input("새 이메일 (없으면 Enter): ")
        address = input("새 주소 (없으면 Enter): ")
        # 빈 입력은 None으로 처리하여야 주소가 없어도 수정이 가능
        update(name, phone if phone else None, email if email else None, address if address else None)
    elif choice == "5":
        show_all() #모든 키-값 을 올린다.
    elif choice == "6":
        address_book ={"address_book":1}
        address_book.clear()
        print(address_book)
        print("모든 값이 삭제 되었습니다.") #클리어를 사용하여 모든 항목을 삭제함
    elif choice == "0":
        print("주소록 프로그램을 종료합니다.")
        break
    else:
        print("[오류] 잘못된 선택입니다.")