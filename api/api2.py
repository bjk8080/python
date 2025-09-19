from flask import Flask
app=Flask(__name__)

def get_menu():
    return{"메뉴": "커피"}
def create_order():
    return {"메세지":"주문 완료"}

app.add_url_rule('/menu', 'get_menu',get_menu, methods=['GET'])
app.add_url_rule('/orders','create_order',create_order, methods=['POST'])

if __name__ == "__main__":
    app.run(debug=True)
