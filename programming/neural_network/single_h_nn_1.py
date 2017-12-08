#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
neural network with 1 hidden layer, 1 input layer, 1 output layer

train the neural network with the whole training samples X,
calculate with matrix instead of vector.

note:
    Judge the sequence when multiplying matrix with dimension needed.

"""
import numpy as np


class SingleHiddenNW(object):
    def __init__(self, eta_1, eta_2=None, hidden_num=None, iter_num=10000):
        self.eta_1 = eta_1
        self.eta_2 = eta_2 if eta_2 else eta_1
        self.iter_num = iter_num
        self.hidden_num = hidden_num

    def fit(self, X, y):
        # todo: reuse output data of hidden layer in bp
        # set neutral number of input and output layer
        self.input_num = X.shape[1]
        self.output_num = y.shape[1]

        # weight of hidden layer, if not provided as parameter, same as input
        if self.hidden_num is None:
            self.hidden_num = X.shape[1]
        # weight of input layer
        # d * q matrix, d is the feature number of x,
        # q is the number of neutral in hidden layer
        np.random.seed(1)
        self.weight_v = 2 * np.random.random((X.shape[1], self.hidden_num)) - 1

        # todo: average zero?
        # q * l, q is the number of neutral in hidden layer,
        # l is the number of output number
        np.random.seed(1)
        self.weight_w = 2 * np.random.random((self.hidden_num, self.output_num)) - 1

        # todo: check whether the num is contradict with each other

        for _ in range(self.iter_num):
            # calculate output
            output = self.predict(X)

            # update weight
            # wj, delta_wj = eta*gj*bh
            bh = self.hidden_output(X)
            bh_trans = bh.T

            gj = self.gj(y, output)

            eh = self.eh(bh, gj)  # l1_delta
            x_trans = X.T

            # multiply should get an matrix (g1, g2 ... gl).T X (b1, b2, ... bh)
            # gj_bh_tmp = np.dot(gj_trans, bh_matrix)
            gj_bh_tmp = np.dot(bh_trans, gj)  # l1.T.dot(l2_delta)
            delta_wj = self.eta_1 * gj_bh_tmp

            delta_theta_j = -1 * self.eta_1 * gj

            # eh_x_tmp = np.dot(eh_trans, x_matrix)
            eh_x_tmp = np.dot(x_trans, eh)
            delta_v_ih = self.eta_2 * eh_x_tmp

            delta_gama = -1 * self.eta_2 * eh
            self.weight_w += delta_wj
            self.weight_v += delta_v_ih

        return self

    def hidden_output(self, x):
        """
        calculate output of hidden layer, vector.

        [b1, b2 ..., bh ..., bq]
        :param x: 1 row of input matrix, 1 sample.
        :return: 1 vector, [b1, b2 ..., bh ..., bq]
        """
        bh_tmp = np.dot(x, self.weight_v)
        bh = self.no_linear_f(bh_tmp)
        # return np.array(bh)
        return bh

    def predict(self, x):
        bh = self.hidden_output(x)
        # matrix, all of bh is already calculated.
        yj_tmp = np.dot(bh, self.weight_w)
        yj = self.no_linear_f(yj_tmp)
        return yj

    def no_linear_f(self, x):
        """
        sigmoid function, 1 / (1 + exp(-x))
        :param x:
        :return:
        """
        return 1.0 / (1.0 + np.exp(-x))

    def gj(self, yj, output_j):
        # l2_delta
        # yj and output_j are vectors
        # gj = output_j(1-output_j)(yj-output_j)
        res = output_j * (1 - output_j) * (yj - output_j)
        return res

    def eh(self, bh, gj):
        # bh(1-hb)*sum(w_hj * gj)
        w_hj = self.weight_w
        # l1_error = l2_delta.dot(syn1.T)
        # l2_delta = gj
        # tmp = np.dot(w_hj, gj_trans)  # sum is in matrix multiply
        tmp = np.dot(gj, w_hj.T)  # l1_error
        res = bh*(1-bh)*tmp  # l1_delta
        return res


if __name__ == "__main__":
    # input data set
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    # output data set
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    nw = SingleHiddenNW(eta_1=0.1, eta_2=0.1, iter_num=600000)

    nw.fit(X, y)

    print(nw.predict([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]))





