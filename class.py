#도서 대출 프로그램
class Book:
    __existing_books = [
        ("메이플", "넥슨", "12345", 1997),
    ]

    def __init__(self, title, author, isbn, year):
        self.__title = title
        self.__author = author
        self.__isbn = isbn
        self.__year = year
        self.__is_borrowed = False

    def get_title(self):
        return self.__title

    def get_author(self):
        return self.__author

    def get_isbn(self):
        return self.__isbn

    def get_year(self):
        return self.__year

    def is_borrowed(self):
        return self.__is_borrowed

    def borrow(self):
        if not self.__is_borrowed:
            self.__is_borrowed = True
            return True
        return False

    def return_book(self):
        if self.__is_borrowed:
            self.__is_borrowed = False
            return True
        return False

    def __str__(self):
        return f"[{self.__year}] title:{self.__title} - author:{self.__author} (ISBN:{self.__isbn})"

    @classmethod
    def get_default_books(cls):
        return [Book(t, a, i, y) for t, a, i, y in cls.__existing_books]


class Member:
    def __init__(self, name, gender, phone):
        self.__name = name
        self.__gender = gender
        self.__phone = phone
        self.__borrowed_books = []

    def get_name(self):
        return self.__name

    def get_gender(self):
        return self.__gender

    def get_phone(self):
        return self.__phone

    def get_borrowed_books(self):
        return self.__borrowed_books

    def borrow_book(self, book: Book):
        if book.borrow():
            self.__borrowed_books.append(book)
            return True
        return False

    def return_book(self, book: Book):
        if book in self.__borrowed_books and book.return_book():
            self.__borrowed_books.remove(book)
            return True
        return False

    def __str__(self):
        return f"name:[{self.__name}] gender:{self.__gender} phone:{self.__phone}"


class Library:
    def __init__(self):
        self.__books = Book.get_default_books()
        self.__members = []

    def add_book(self, book: Book):
        self.__books.append(book)
        print(f"[추가 완료] {book.get_title()} 도서가 추가되었습니다.")

    def delete_book(self, title: str):
        for book in self.__books:
            if book.get_title() == title:
                self.__books.remove(book)
                print(f"[삭제완료] {book.get_title()} 도서가 삭제되었습니다.")
                return
        print("삭제할 도서를 찾을 수 없습니다.")

    def find_book(self, keyword: str):
        found = False
        for b in self.__books:
            if b.get_title() == keyword or b.get_author() == keyword or b.get_isbn() == keyword:
                print(f"[검색결과] {b}")
                found = True
        if not found:
            print("찾는 도서를 검색할 수 없습니다.")

    def all_delete_book(self):
        self.__books.clear()
        print("모든 도서가 삭제되었습니다.")

    def show_books(self):
        if not self.__books:
            print("현재 도서관에 등록된 책이 없습니다.")
        else:
            print("\n=== 도서 목록 ===")
            for book in self.__books:
                print(book)

    def add_member(self, member: Member):
        self.__members.append(member)
        print(f"[추가 완료] {member.get_name()} 님이 추가되었습니다.")
    
    def delete_member(self, member_name: str):
        found = False
        for member in self.__members:
                if member.get_name() == member_name:
                    self.__members.remove(member)
                    print(f"[삭제완료] {member.get_name()} 님이 삭제되었습니다.")
                    found = True
                    break
        if not found:
            print(f"[오류] {name}님이 존재 하지 않습니다.")
                


    def show_members(self):
        if not self.__members:
            print("등록된 회원이 없습니다.")
        else:
            print("\n=== 회원 목록 ===")
            for member in self.__members:
                print(member)

    def borrow_book(self, member_name: str, book_title: str):
        member = next((m for m in self.__members if m.get_name() == member_name), None)
        if not member:
            print("[오류] 해당 회원이 없습니다.")
            return

        for book in self.__books:
            if book.get_title() == book_title:
                if member.borrow_book(book):
                    print(f"[대출완료] '{member.get_name()} 님이 {book.get_title()}' 도서를 대출했습니다.")
                else:
                    print(f"[대출불가] '{book.get_title()}' 은/는 이미 대출 중입니다.")
                return
        print("[오류] 해당 도서를 찾을 수 없습니다.")

    def return_book(self, member_name: str, book_title: str):
        member = next((m for m in self.__members if m.get_name() == member_name), None)
        if not member:
            print("[오류] 해당 회원이 없습니다.")
            return

        for book in member.get_borrowed_books():
            if book.get_title() == book_title:
                if member.return_book(book):
                    print(f"[반납완료] '{member.get_name()} 님이 {book.get_title()}' 도서를 반납했습니다.")
                return
        print("[오류] 해당 회원이 해당 도서를 빌린 기록이 없습니다.")

    def show_borrowed_books(self):
        any_borrowed = False
        for member in self.__members:
            if member.get_borrowed_books():
                print(f"\n[회원] {member.get_name()} 님이 대출한 도서:")
                for book in member.get_borrowed_books():
                    print(f"  - {book.get_title()} / {book.get_author()} (ISBN:{book.get_isbn()})")
                any_borrowed = True
        if not any_borrowed:
            print("현재 대출 중인 도서가 없습니다.")

# 파일 실행기

library = Library()

while True:
    print("\n===== 도서관 메뉴 =====")
    print("1. 도서 추가")
    print("2. 선택 도서 삭제")
    print("3. 모든 도서 삭제")
    print("4. 도서 목록 보기")
    print("5. 회원 추가")
    print("6. 회원 목록")
    print("7. 회원 삭제")
    print("8. 도서 검색")
    print("9. 도서 대출")
    print("10. 도서 반납")
    print("11. 도서 대출 현황")
    print("0. 종료")
    
    choice = input("선택: ")
    
    if choice == "1":
        title = input("제목: ")
        author = input("저자: ")
        isbn = input("ISBN: ")
        year = input("출판연도: ")
        
        book = Book(title, author, isbn, year)  
        library.add_book(book)  
    elif choice == "2":
        title = input("삭제할 도서 제목: ")
        library.delete_book(title)
    elif choice == "3":
        library.all_delete_book()
    elif choice == "4":
        library.show_books()
    elif choice == "5":
        name = input("이름: ")
        gender = input("성별: ")
        phone = input("전화번호: ")
        
        member=Member(name, gender, phone)
        library.add_member(member)
    elif choice =="6":
        library.show_members()
    elif choice == "7":
        name=input("삭제할 회원 명을 입력하세요: ")
        library.delete_member(name)
    elif choice == "8":
        keyword = input("제목 / 저자 / ISBN 중 하나를 입력하세요: ")
        library.find_book(keyword)
    elif choice == "9":
        member_name = input("회원 이름: ")
        book_title = input("대출할 도서 제목: ")
        library.borrow_book(member_name, book_title)
    elif choice == "10":
        member_name = input("회원 이름: ")
        book_title = input("반납할 도서 제목: ")
        library.return_book(member_name, book_title)
    elif choice == "11":
        library.show_borrowed_books()

    elif choice == "0":
        print("프로그램을 종료합니다.")
        break
    
    else:
        print("[오류] 잘못된 선택입니다.")