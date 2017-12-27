# coding:utf-8
import numpy as np
from sklearn.model_selection import train_test_split
import RNN


def _main():
    prob_filename = 'sin-cos-prob.npy'

    # パラメータ
    maxlen = 25
    n_hidden = 20
    epochs = 500
    batch_size = 10

    # 学習用データを準備
    problem_list = np.load(prob_filename)
    length_of_sequences = len(problem_list)
    data = []
    target = []
    for i in range(0, length_of_sequences - maxlen):
        data.append(problem_list[i: i+maxlen])
        target.append(problem_list[i + maxlen])
    X = np.array(data).reshape(len(data), maxlen, 2)
    Y = np.array(target).reshape(len(data), 2)
    n_train = int(len(data) * 0.9)
    n_validation = len(data) - n_train
    x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=n_validation)
    n_in = len(X[0][0])
    n_out = len(Y[0])

    # 学習
    model = RNN.RNN(n_in=n_in, n_hiddens=n_hidden, n_out=n_out, maxlen=maxlen, data_name=prob_filename[:-3])
    model.fit(x_train=x_train, y_train=y_train, x_validation=x_validation, y_validation=y_validation,
             batch_size=batch_size, epochs=epochs)

    # 予測用のデータを準備
    answer_list = np.load('sin-cos-ans.npy')
    precedence = np.array([answer_list[0:maxlen]])
    predicted = [[None, None] for i in range(maxlen)]

    # 予測
    model.predict(precedence, predicted, length_of_sequences)
    # 予測結果を描画
    model.draw_epochs()
    # loss遷移を描画
    model.draw_prediction(answer_list, predicted)


if __name__ == '__main__':
    _main()
