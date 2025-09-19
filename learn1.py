from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class DigitClassifier:
    def __init__(self):
        self.W = np.random.randn(64, 10) * 0.01  # 가중치 행렬 초기화 , (8*8=64, 클래스가 10개이므로 10)
        self.b = np.zeros(10)  # 편향 벡터 초기화

    def softmax(self, z): #모델의 출력값을 확률로 변환시키는 것
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) #z값에서 z최대값을 뺀 뒤 지수처리 -> 오버플로우 방지 why? e의 배수는 점점 커지면 컴퓨터로도 연산이 어려움
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # 정규분포로 변환 한번더 공부

    def cross_entropy_loss(self, y_pred, y_true): #y_pred 모델의 예측값 y_true 실제 값, 두개를 비교해서 손실을 예측하는 함수
        m = y_true.shape[0] # 샘플의 개수를 구하는 공식
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
         # np.log는 자연로그를 계산, e-9를 더하는 이유는 0이 되는 경우를 방지, 모든 클래스를 더한 후 나눠 클래스의 손실 평균을 구함
        return loss

    def forward(self, x):
        z = x @ self.W + self.b # @로 행렬을 곱한다,가중치를 행렬로 곱을 한 뒤 상수값을 더한다. 상수를 더하는 이유는 모델이 데이터를 자유롭게 맞추기 위한 값이다. 즉 self.b는 행렬의 곱을 이동 시키게 하는 벡터
        return self.softmax(z) # 내가 구한 로짓 값을 확률로 변환시켜라

    def backward(self, x, y): # 역전파 
        m = y.shape[0]
        y_pred = self.forward(x) #foward 함수를 호출하여 예측값을 구함
        d_loss_dy_pred = y_pred - y # 손실값을 모델 예측값- 실제 예측값
        dW = (x.T @ d_loss_dy_pred) / m # 가중치의 기울기 구하기, 각 입력 픽셀이 손실에 얼마나 영향을 끼쳤는지 알아야 가중치값을 변화시킬 수 있음
        db = np.sum(d_loss_dy_pred, axis=0) / m # 편향 b의 기울기, 특정 숫자에 대해 낮은 확률 값을 조정하는데 사용
        return dW, db #dw,db 를 구해야 잘못 예측한 값을 깨닫고 수정하는 방법을 깨달음

    def predict(self, x):
        y_pred_proba = self.forward(x) 
        return np.argmax(y_pred_proba, axis=1) #각 행 별 가장 큰값의 인덱스를 뽑아줌

    def accuracy(self, x, y):
        y_pred = self.predict(x) # 모델이 예측한 클래스 인덱스를 가져옴
        y_true = np.argmax(y, axis=1) # 원-핫 라벨 행에서 장답 클래스 인덱스를 뽑기
        return np.mean(y_pred == y_true) # 예측과 정답이 일치한 비율을 계산

    def train(self, x_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.1):
        # epochs는 반복 횟수 , learning_rate 가중치 갱신 (하이퍼 파라미터)
        losses = []
        train_accs = []
        test_accs = []
        #손실, 훈련 정확도, 테스트 정확도를 저장하여 분석
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(x_train) # x_train에 대해 예측 확률을 계산

            # 손실 계산
            loss = self.cross_entropy_loss(y_pred, y_train) # 전체 데이터에 대한 평균 손실을 계산
            losses.append(loss)

            # 역전파로 기울기 계산
            dW, db = self.backward(x_train, y_train)# 현재의 w,b의 값으로 x,y_train을 사용해 기울기 계산

            # 가중치 업데이트
            self.W -= learning_rate * dW #가중치 값을 현재의 데이터로 조정하여 손실을 줄임
            self.b -= learning_rate * db

            # 훈련/테스트 정확도
            train_acc = self.accuracy(x_train, y_train)
            test_acc = self.accuracy(x_test, y_test)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Loss={loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        return losses, train_accs, test_accs
def mini_mnist():
    """
    8x8 숫자 이미지 분류 (sklearn digits 데이터셋 사용)
    """

    # 데이터 불러오기
    digits = load_digits()
    x, y = digits.data, digits.target
    x = x / 16.0  # 정규화 (픽셀 값: 0~16 → 0~1)

    # 원-핫 인코딩
    y_onehot = np.eye(10)[y]

    # train/test 분리
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_onehot, test_size=0.1, random_state=42
    )

    print("미니 MNIST 과제")
    print(f"훈련 데이터: {x_train.shape}")
    print(f"테스트 데이터: {x_test.shape}")
    print(f"클래스 수: 10개 (0-9 숫자)")

    # 모델 생성
    clf = DigitClassifier()

    # 학습
    losses, train_accs, test_accs = clf.train(
        x_train, y_train, x_test, y_test,
        epochs=1000, learning_rate=0.85
    ) # epochs 학습 횟수 , learning_rate 학습 속도

    # 최종 정확도 출력
    print(f"최종 훈련 정확도: {train_accs[-1]*100:.2f}%")
    print(f"최종 테스트 정확도: {test_accs[-1]*100:.2f}%")

    # 클래스별 정확도 계산
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = clf.predict(x_test)
    num_classes = 10
    print("각 숫자별 테스트 정확도:")
    for i in range(num_classes):
        idx = (y_test_labels == i)
        correct = np.sum(y_pred_labels[idx] == y_test_labels[idx])
        total = np.sum(idx)
        acc = correct / total * 100
        print(f"숫자 {i}: {correct}/{total} 정확도 {acc:.2f}%")

    # 손실 시각화
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    mini_mnist()