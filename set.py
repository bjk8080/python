#사용자들의 취미
users = {
    "병직": {"음악","영화","사격"},
    "지우": {"음악","게임","식사"},
    "승준": {"커피","게임","영화"},
    "태수": {"사육","운동","산책"},
}

# 친구 취미 교집합
def recommend(user_name):
    if user_name not in users: #추천을 원하는 인원에 이름이 없을 시 없다고 생성
        print(f"[오류] {user_name} 사용자가 존재하지 않습니다.")
        return
    
    user_hobbies = users[user_name] #취미를 유저 집합과 연결
    same = [] #공통 관심사
    different = [] #비공통 관심사
    
    for other, hobbies in users.items(): #모든 사용자들을 비교
        if other == user_name: #자기 자신은 제외
            continue
        if user_hobbies & hobbies:  # 교집합이 존재하면 공통 관심사 있음
            same.append(other)
        else: #없으면 공통 관심사가 없는 친구
            different.append(other)
    
    print(f"\n[{user_name}님의 추천 친구]")
    print(f"공통 관심사가 있는 친구: {same if same else '없음'}")
    print(f"공통 관심사가 없는 친구: {different if different else '없음'}")

# 테스트
recommend("병직")
recommend("지우")
recommend("승준")
recommend("태수")
