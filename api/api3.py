from fastapi import FastAPI

app = FastAPI()

@app.get("/menu")
def get_menu():
    return{"메뉴":["아메리카노","라떼","카푸치노"]}

@app.post("/orders")
def create_order(order_data: dict):
    return {"메세지":"주문 완료","부문번호":123}

@app.put("/orders/123")
def update_order(order_data:dict):
    return {"메세지":"주문이 수정되었습니다."}

@app.patch("/orders/123")
def patch_order(partial_data:dict):
    return{"메세지": "주문 옵션이 변경되었습니다."}

@app.delete("/orders/123")
def cancel_order():
    return{"메세지": "주문이 취소되었습니다."}

