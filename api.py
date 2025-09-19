import asyncio
import aiohttp
import time
import concurrent.futures
import random

websites =[
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5"
]

def task(name):
    print(f"wkrdjq{name} 시작")
    delay=random.uniform(0.5,2)
    time.sleep(delay)
    print(f"작업{name}완료(소요시간: {delay:.2f}초)")
    content_length = random.randint(200, 300) 
    return content_length
params=[
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5"
]
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as executor:
    results=list(executor.map(task,params))
    for result in results:
        print(result)
async def fetch(session, url):
    print(f"{url} 요청시작")
    try:
        start_time = time.time()
        async with session.get(url, timeout=10) as response:
            content=await response.text()
            elapsed=time.time() - start_time
            print(f"{url} 응답 완료: {len(content)}")
            return url, len(content), elapsed
    except Exception as e:
        print(f"{url} 오류 발생: {e}")
        return url, 0,0
async def fetch_all_sequential(urls):
    start_time=time.time()
    results=[]

    async with aiohttp.ClientSession() as session:
        for url in urls:
            result = await fetch(session,url)
            results.append(result)

    end_time=time.time()
    print(f"순차 처리 완료: {end_time -start_time:.2f}초 소요")
    return results
async def fetch_all_parallel(urls):
    start_time=time.time()
    async with aiohttp.ClientSession() as session:
        tasks=[fetch(session,url)for url in urls]
        results = await asyncio.gather(*tasks)
    
    end_time=time.time()
    print(f"병렬 처리 완료: {end_time-start_time:.2f}초 소요")
    return results

async def main():
    print("\n=== 순차 처리 시작 ===")
    sequential_results=await fetch_all_sequential(websites)
    await asyncio.sleep(1)
    print("\n===병렬 처리 시작===")
    parallel_results=await fetch_all_parallel(websites)
    seq_total_bytes=sum(r[1]for r in sequential_results)
    par_total_bytes=sum(r[1]for r in parallel_results)

    thread_results = sum(results)
    print("===결과 요약===")
    print("순차 처리", seq_total_bytes)
    print("병렬 처리", par_total_bytes)
    print("Thread 처리", thread_results)

if __name__ == "__main__":
    asyncio.run(main())