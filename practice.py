from typing import List, Optional


class Book:
    """도서 정보를 관리하는 클래스"""
    def __init__(self, title: str, author: str, isbn: str, year: int):
        self._title = title
        self._author = author
        self._isbn = isbn
        self._year = year
        self._is_borrowed = False  # 대출 여부

    # Getter (캡슐화된 정보 접근)
    @property
    def title(self):
        return self._title

    @property
    def author(self):
        return self._author

    @property
    def isbn(self):
        return self._isbn

    @property
    def year(self):
        return self._year

    @property
    def is_borrowed(self):
        return self._is_borrowed

    def borrow(self):
        if self._is_borrowed:
            raise Exception(f"이미 대출된 도서입니다: {self._title}")
        self._is_borrowed = True

    def return_book(self):
        self._is_borrowed = False

    def __str__(self):
        status = "대출중" if self._is_borrowed else "대출가능"
        return f"[{self._year}] {self._title} / {self._author} (ISBN:{self._isbn}, {status})"


class Member:
    """도서관 회원 관리"""
    def __init__(self, name: str, member_id: str):
        self._name = name
        self._member_id = member_id
        self._borrowed_books: List[Book] = []

    @property
    def name(self):
        return self._name

    @property
    def member_id(self):
        return self._member_id

    def borrow_book(self, book: Book):
        if book.is_borrowed:
            raise Exception("이 책은 이미 대출되었습니다.")
        book.borrow()
        self._borrowed_books.append(book)

    def return_book(self, book: Book):
        if book not in self._borrowed_books:
            raise Exception("이 회원이 대출한 책이 아닙니다.")
        book.return_book()
        self._borrowed_books.remove(book)

    def borrowed_books(self):
        return [str(book) for book in self._borrowed_books]

    def __str__(self):
        return f"회원 {self._name} (ID: {self._member_id})"


class Library:
    """도서관 전체 관리"""
    def __init__(self):
        self._books: List[Book] = []
        self._members: List[Member] = []

    # 도서 관련
    def add_book(self, book: Book):
        self._books.append(book)

    def remove_book(self, isbn: str):
        self._books = [book for book in self._books if book.isbn != isbn]

    def search_by_title(self, title: str) -> List[Book]:
        return [book for book in self._books if title.lower() in book.title.lower()]

    def search_by_author(self, author: str) -> List[Book]:
        return [book for book in self._books if author.lower() in book.author.lower()]

    def search_by_isbn(self, isbn: str) -> Optional[Book]:
        for book in self._books:
            if book.isbn == isbn:
                return book
        return None

    # 회원 관련
    def register_member(self, member: Member):
        self._members.append(member)

    def get_member(self, member_id: str) -> Optional[Member]:
        for member in self._members:
            if member.member_id == member_id:
                return member
        return None

    # 대출/반납
    def borrow_book(self, member_id: str, isbn: str):
        member = self.get_member(member_id)
        book = self.search_by_isbn(isbn)
        if not member:
            raise Exception("회원이 존재하지 않습니다.")
        if not book:
            raise Exception("책이 존재하지 않습니다.")
        member.borrow_book(book)

    def return_book(self, member_id: str, isbn: str):
        member = self.get_member(member_id)
        book = self.search_by_isbn(isbn)
        if not member:
            raise Exception("회원이 존재하지 않습니다.")
        if not book:
            raise Exception("책이 존재하지 않습니다.")
        member.return_book(book)

    def show_all_books(self):
        return [str(book) for book in self._books]

    def show_member_status(self, member_id: str):
        member = self.get_member(member_id)
        if not member:
            return "회원이 존재하지 않습니다."
        return member.borrowed_books()


# ==========================
# 사용 예시
# ==========================
if __name__ == "__main__":
    library = Library()

    # 도서 등록
    book1 = Book("해리포터", "J.K. 롤링", "12345", 1997)
    book2 = Book("반지의 제왕", "톨킨", "67890", 1954)
    library.add_book(book1)
    library.add_book(book2)

    # 회원 등록
    member1 = Member("홍길동", "M001")
    library.register_member(member1)

    # 대출
    library.borrow_book("M001", "12345")
    print("회원 대출 현황:", library.show_member_status("M001"))

    # 반납
    library.return_book("M001", "12345")
    print("반납 후 현황:", library.show_member_status("M001"))
