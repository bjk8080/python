import threading
import time
import queue
import random


task_queue=queue.Queue()
result_queue=queue.Queue()

def create_tasks():
    print("작업 생성 시작")
    for i in range(10):
        task = f"작업-{i}"
        task_queue.put(task)
        print(f"작업 '{task}'추가됨")
        time.sleep(random.uniform(0.1,0.3))
    for_in range(3):
        task_queue.put(None)
    print('모든 작업 생성 완료')

def worker(worker_id):
    print(f"워커 : {worker_id}시작")
    while True:
        task=task_queue.get()
        if task is None:
            print(f"워커: {worker_id} 작업종료")
            break

        #작업 코드

        #작업 완료
        result_queue.put((worker_id, result))
        task_queue.task_done()
        print(f"남은 작업 수 : {task_queue.qsize()}")
        def result_dollecotr():
            print("결과 수집기 시작")
            results=[]
            for_in range(10):
                worker_id, result=result_queue.get()
                print(f"결과 수집 워커{worker_id} -> {result}")
                results.append(result)
                result_queue
creator=threading.Thread(target=create_tasks)
creator=[threading.Thread(target=workers,)]
