import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras import models, layers
import seaborn as sns

from plot_util import plot_acc, plot_loss


# 코드 실행 및 결과 보기
def main():
    machine = Machine()
    machine.run(epochs=400)

class Machine():
    def __init__(self):
        self.data = Dataset()
        shape = self.data.X.shape[1:]
        self.model = rnn_model(shape)

    def run(self, epochs=400):
        d = self.data
        X_train, X_test = d.X_train, d.X_test
        y_train, y_test = d.y_train, d.y_test

        X, y = d.X, d.y

        m = self.model
        history = m.fit(X_train, y_train, epochs=epochs, validation_data = [X_test, y_test], verbose = 0)

        plot_loss(history)
        plt.title('History of training')
        plt.show()

        yp = m.predict(X_test)
        print('Loss:', m.evaluate(X_test, y_test))
        plt.plot(yp, label='Original')
        plt.plot(y_test, label='Prediction')
        plt.legend(loc = 0)
        plt.title('Validation Results')
        plt.show()

        yp = m.predict(X_test).reshape(-1)
        print('Loss:', m.evaluate(X_test, y_test))
        print(yp.shape, y_test.shape)

        df = pd.DataFrame()
        df['Sample'] = list(range(len(y_test))) * 2
        df['Normalized #Passengers'] = np.concatenate([y_test, yp], axis = 0)
        df['Type'] = ['Original'] * len(y_test) + ['Prediction'] * len(yp)

        plt.figure(figsize=(7, 5))
        sns.barplot(x="Sample", y = "Normalized #Passengers", hue="Type", data=df)

        plt.ylabel('Normalized #Passengers')
        plt.show()

        yp = m.predict(X)

        plt.plot(yp, label='Original')
        plt.plot(y, label='Prediction')
        plt.legend(loc = 0)
        plt.title('All Results')
        plt.show()


# LSTM 시계열 회귀 모델링
def rnn_model(shape):
    m_x = layers.Input(shape=shape)
    m_h = layers.LSTM(10)(m_x)
    m_y = layers.Dense(1)(m_h)
    m = models.Model(m_x, m_y)

    m.compile('adam', 'mean_squared_error')

    m.summary()

    return m

# 데이터 불러오기
class Dataset:
    def __init__(self, fname='international-airline-passengers.csv', D = 12):
        data_dn = load_data(fname=fname)
        X, y = get_Xy(data_dn, D=D)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 42) # random_state = seed

        self.X, self.y = X, y
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

def load_data(fname='international-airline-passengers.csv'):
    dataset = pd.read_csv(fname, usecols=[1], engine='python', skipfooter=3)
    data = dataset.values.reshape(-1) # (m, 1) -> (m,)

    plt.plot(data)
    plt.xlabel('Time'); plt.ylabel('#Passengers')
    plt.title('Original Data')
    plt.show()

    # data normalize
    data_dn = (data - np.mean(data)) / np.std(data) / 5

    plt.plot(data_dn)
    plt.xlabel('Time'); plt.ylabel('#Passengers')
    plt.title('Normalized Data by $E[]$ and $5\sigma$')
    plt.show()

    return data_dn

# D개월 간의 승객 수 변화를 통해 그다음 달의 승객 수를 예측할 수 있는지 알아보는 데이터셋 생성
def get_Xy(data, D=12):
    # make X and y
    X_l = []
    y_l = []
    N = len(data)

    assert N > D, "N should be larger than D, where N is len(data)"

    for ii in range(N-D-1):
        X_l.append(data[ii:ii+D])
        y_l.append(data[ii+D])

    X = np.array(X_l)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y_l)

    print(X.shape, y.shape) # (m, 12, 1) (m,)
    return X, y

if __name__ == "__main__":
    main()

'''
(131, 12, 1) (131,)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 12, 1)             0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 10)                480       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11        
=================================================================
Total params: 491
Trainable params: 491
Non-trainable params: 0
_________________________________________________________________
'''