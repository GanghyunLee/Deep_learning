# Regression ANN 모델링

from keras import layers, models

class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        
        x = layers.Input(shape=(Nin, ))
        h = relu(hidden(x))
        y = output(h)       # Regression 문제이므로 Activation Function = Linear Activation
        
        super().__init__(x, y)
        self.compile(loss='mse', optimizer='sgd') # Regression 문제이므로 loss function = Mean square error
        
# 학습과 평가용 데이터 불러오기
from keras import datasets
from sklearn import preprocessing

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    # 데이터를 0~1로 정규화
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return (X_train, y_train), (X_test, y_test)

# 회귀 ANN 학습 결과 그래프 구현
import matplotlib.pyplot as plt
from plot_util import plot_loss

# 회귀 ANN 학습 및 성능 분석
def main():
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()

    history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_split=0.2, verbose = 2)

    performance_test = model.evaluate(X_test, y_test, batch_size = 100)
    print('\nTest Loss -> {:.2f}'.format(performance_test))

    plot_loss(history)
    plt.show()

if __name__ == '__main__':
    main()

