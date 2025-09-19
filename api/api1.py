from fastapi import FastAPI

app=FastAPI()

@app.get("/")
def read_root():
    return {"message":"안녕하세요 이디야 입니다!"}

@app.get("/hello")
def say_hello():
    return{"greeting":"Hello","cafe": "ideya"}