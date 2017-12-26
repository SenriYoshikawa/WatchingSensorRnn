import tensorflow as tf
import EarlyStopping
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

class RNN(object):
    def __init__(self, n_in, n_hiddens, n_out, maxlen, data_name):
        self._n_in = n_in
        self._n_hidden = n_hiddens
        self._n_out = n_out
        self._maxlen = maxlen
        self.early_stopping = EarlyStopping.EarlyStopping(patience=10, verbose=1)
        self._x = None
        self._y = None
        self._t = None
        self._n_batch = None
        self._sess = None
        self._data_name = data_name
        self._history = {
            'loss': [],
            'accuracy': []
        }

    @staticmethod
    def __weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def __bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    def inference(self, x, n_batch):
        cell = tf.contrib.rnn.GRUCell(self._n_hidden)
        initial_state = cell.zero_state(n_batch, tf.float32)

        state = initial_state
        outputs = []
        with tf.variable_scope('RNN'):
            for t_i in range(self._maxlen):
                if t_i > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:, t_i, :], state)
                outputs.append(cell_output)

        output = outputs[-1]

        v = self.__weight_variable([self._n_hidden, self._n_out])
        c = self.__bias_variable([self._n_out])
        y_i = tf.matmul(output, v) + c
        return y_i

    @staticmethod
    def training(loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        train_step = optimizer.minimize(loss)
        return train_step

    @staticmethod
    def __loss(y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    @staticmethod
    def __accuracy(y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def fit(self, x_train, y_train, x_validation, y_validation, epochs=500, batch_size=100):
        print('learn start')
        x = tf.placeholder(tf.float32, shape=[None, self._maxlen, self._n_in])
        t = tf.placeholder(tf.float32, shape=[None, self._n_out])
        n_batch = tf.placeholder(tf.int32, [])

        # evaluation用に保持
        self._x = x
        self._t = t

        self._n_batch = n_batch
        y = self.inference(x=x, n_batch=n_batch)
        loss = self.__loss(y, t)
        accuracy = self.__accuracy(y, t)
        train_step = self.training(loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            # evaluate()用に保持
            self._sess = sess

            n_train = len(x_train)
            n_validation = len(x_validation)
            n_batches = n_train // batch_size

            for epoch in range(epochs):
                X_, Y_ = shuffle(x_train, y_train)

                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size

                    sess.run(train_step, feed_dict={
                        x: X_[start:end],
                        t: Y_[start:end],
                        n_batch: batch_size
                    })

                loss_ = loss.eval(session=sess, feed_dict={
                    x: x_validation,
                    t: y_validation,
                    n_batch: n_validation
                })

                accuracy_ = accuracy.eval(session=sess, feed_dict={
                    x: x_validation,
                    t: y_validation,
                    n_batch: n_validation
                })

                self._history['loss'].append(loss_)
                self._history['accuracy'].append(accuracy_)
                print('epoch;', epoch, '  loss:', loss_, ' accuracy:', accuracy_)

                if self.early_stopping.validate(loss_):
                    break
            self._y = y
            saver = tf.train.Saver()
            cwd = os.getcwd()
            if os.path.exists(self._data_name):
                shutil.rmtree('./' + self._data_name)
            os.makedirs(self._data_name)
            saver.save(sess, cwd + '\\' + self._data_name + '\model.ckpt')

        return self._history

    def predict(self, Z, predicted, length_of_sequences):
        print('predict start')

        if self._x is None:
            x = tf.placeholder(tf.float32, shape=[None, self._maxlen, self._n_in])
            n_batch = tf.placeholder(tf.int32, [])
            y = self.inference(x=x, n_batch=n_batch)
        else:
            x = self._x
            y = self._y
            n_batch = self._n_batch

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            cwd = os.getcwd()
            saver.restore(sess, cwd + '\\' + self._data_name + '\model.ckpt')

            for i in range(length_of_sequences - self._maxlen):
                z_ = Z[-1:]
                y_ = y.eval(session=sess, feed_dict={
                    x: Z[-1:],
                    n_batch: 1
                })

                sequence_ = np.concatenate((z_.reshape(self._maxlen, self._n_in)[1:], y_), axis=0).reshape(1, self._maxlen,
                                                                                                           self._n_in)
                Z = np.append(Z, sequence_, axis=0)
                predicted.append(y_.reshape(-1))

        predicted = np.array(predicted)
        return predicted

    def draw_epochs(self):
        plt.cla()
        plt.figure()
        plt.plot(range(len(self._history['loss'])), self._history['loss'], label='loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.savefig("history.png")
        print("saved to history.png")

    @staticmethod
    def draw_prediction(original, predicted):
        predicted = np.array(predicted)
        #plt.rc('font', family='serif')
        plt.cla()
        plt.figure()
        plt.plot(original[:, 0], linestyle='dashed', color='gray')
        plt.plot(predicted[:, 0])
        plt.plot(original[:, 1], linestyle='dashed', color='gray')
        plt.plot(predicted[:, 1])
        plt.savefig("sin test.png")
        print("save to sin test.png")

    '''
    def evaluate(self, x_test, y_test):
        return self.accuracy.eval(session=self._sess, feed_dict={
            self._x = x_test,
            self._t = y_test,
            self._n_batch:            
        })
    '''