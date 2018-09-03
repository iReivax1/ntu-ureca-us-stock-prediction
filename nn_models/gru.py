from math import isnan
import logging
import numpy as np
import pickle
import os
import tensorflow as tf


LAMDA = 0.01
RATIO = 1.0
PERIOD = 2  # 10
LEARNING_RATE = 0.005
NEPOCH = 100
EVALUATE_LOSS_AFTER = 10


class GRU_2(object):
    def __init__(self, input_dim, hidden_dim, seed):
        self.graph = tf.Graph()
        with self.graph.as_default():
            logging.debug("current seed equals to: %s" % seed)
            # Parameters: Layer 1
            Uz = tf.Variable(tf.random_uniform([hidden_dim, input_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            Ur = tf.Variable(tf.random_uniform([hidden_dim, input_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            U_ = tf.Variable(tf.random_uniform([hidden_dim, input_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))

            Wz = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            Wr = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            W_ = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))

            bz = tf.Variable(tf.zeros([hidden_dim, 1]))
            br = tf.Variable(tf.zeros([hidden_dim, 1]))
            b_ = tf.Variable(tf.zeros([hidden_dim, 1]))

            # Parameters: Layer 2
            Uz2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            Ur2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            U_2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))

            Wz2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            Wr2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            W_2 = tf.Variable(tf.random_uniform([hidden_dim, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))

            bz2 = tf.Variable(tf.zeros([hidden_dim, 1]))
            br2 = tf.Variable(tf.zeros([hidden_dim, 1]))
            b_2 = tf.Variable(tf.zeros([hidden_dim, 1]))

            # Parameters: output
            V = tf.Variable(tf.random_uniform([1, hidden_dim], -np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), seed=seed))
            c = tf.Variable(tf.zeros([1, 1]))

            # Definition of the cell computation.
            def gru_cell(x_t, h_t_prev, h_t_prev2):
                # Layer 1
                z = tf.sigmoid(tf.matmul(Uz, x_t) + tf.matmul(Wz, h_t_prev) + bz)
                r = tf.sigmoid(tf.matmul(Ur, x_t) + tf.matmul(Wr, h_t_prev) + br)
                h_ = tf.tanh(tf.matmul(U_, x_t) + tf.matmul(W_, r * h_t_prev) + b_)
                h = tf.multiply((1 - z), h_) + tf.multiply(z, h_t_prev)

                # Layer 2
                z2 = tf.sigmoid(tf.matmul(Uz2, h) + tf.matmul(Wz2, h_t_prev2) + bz2)
                r2 = tf.sigmoid(tf.matmul(Ur2, h) + tf.matmul(Wr2, h_t_prev2) + br2)
                h_2 = tf.tanh(tf.matmul(U_2, h) + tf.matmul(W_2, r2 * h_t_prev2) + b_2)
                h2 = tf.multiply((1 - z2), h_2) + tf.multiply(z2, h_t_prev2)

                # Output
                output = tf.tanh(tf.matmul(V, h2) + c)[0][0]
                return output, h, h2

            # Input data.
            self.train_data = list()
            for _ in range(PERIOD):
                self.train_data.append(
                    tf.placeholder(tf.float32, shape=[input_dim, 1]))

            train_inputs = self.train_data
            self.train_labels = tf.placeholder(tf.float32)

            # Unrolled GRU loop.
            outputs = list()
            state = tf.Variable(tf.zeros([hidden_dim, 1]), trainable=False)
            state2 = tf.Variable(tf.zeros([hidden_dim, 1]), trainable=False)
            for i in train_inputs:
                output, state, state2 = gru_cell(i, state, state2)
                outputs.append(output)

            self.logits = outputs
            lamda = LAMDA
            self.regulization = lamda * (
                tf.nn.l2_loss(Uz) + tf.nn.l2_loss(Ur) + tf.nn.l2_loss(U_) +
                tf.nn.l2_loss(Wz) + tf.nn.l2_loss(Wr) + tf.nn.l2_loss(W_) +
                tf.nn.l2_loss(br) + tf.nn.l2_loss(bz) + tf.nn.l2_loss(b_) +

                tf.nn.l2_loss(Uz2) + tf.nn.l2_loss(Ur2) + tf.nn.l2_loss(U_2) +
                tf.nn.l2_loss(Wz2) + tf.nn.l2_loss(Wr2) + tf.nn.l2_loss(W_2) +
                tf.nn.l2_loss(br2) + tf.nn.l2_loss(bz2) + tf.nn.l2_loss(b_2) +

                tf.nn.l2_loss(V) + tf.nn.l2_loss(c))

            self.loss = tf.sqrt(tf.squared_difference(self.logits[-1], self.train_labels[0][0])) + self.regulization  # loss的dim: 1*1

            # Optimizer.
            global_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(
                LEARNING_RATE, global_step, 10000, 0.95, staircase=True)
            # self.learning_rate = tf.placeholder(tf.float32, shape=[])
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)  # 防止梯度爆炸
            self.optimizer = optimizer.apply_gradients(
                zip(gradients, v), global_step=global_step)
            self.gradients = gradients
            self.v = v
            # self.optimizer = optimizer.minimize(loss=self.loss)

            # Predictions.
            self.train_prediction = self.logits[-1]  # 最后一列是需要预测的y
            self.saver = tf.train.Saver()


def train_model(model, data, train_y, seed, id=None):  # train_y的维度比data大1

    BGN_LENGTH = np.int(np.ceil(data.shape[0] * RATIO))

    if id is None:
        para = get_para_name('para', seed)
        data_path = os.path.join(os.getcwd(), 'parameters', para + '.ckpt')
        para = get_para_name('pred', seed)
        pickle_path = os.path.join(os.getcwd(), 'parameters', para + '.pic')
    else:
        para = get_para_name('ada' + str(id), seed)
        data_path = os.path.join(os.getcwd(), 'parameters', para + '.ckpt')
        para = get_para_name('pred' + str(id), seed)
        pickle_path = os.path.join(os.getcwd(), 'parameters', para + '.pic')

    with tf.Session(graph=model.graph) as session:

        preds = []
        if os.path.isfile(data_path + '.index'):  # Determine if model was trained
            model.saver.restore(session, data_path)
            with open(pickle_path, 'rb') as f:
                preds = pickle.load(f)
        else:  # 若无训练过则开始训练前BGN_LENGTH天的模型
            tf.set_random_seed(seed)
            tf.global_variables_initializer().run()
            logging.debug(str(seed) + 'Initialized')

            # 用前BGN_LENGTH天的数据训练model
            for epoch in range(NEPOCH):

                for step in range(BGN_LENGTH - PERIOD):

                    feed_dict = dict()
                    for i in range(PERIOD):  # 0 ~ 10
                        feed_dict[model.train_data[i]] = np.reshape(data[step + i], (data.shape[1], 1))
                    feed_dict[model.train_labels] = np.reshape(train_y[step + PERIOD], (1, 1))

                    _, v, ls = session.run([model.optimizer, model.v, model.loss], feed_dict=feed_dict)
                    if isnan(ls):
                        # print("epoch %s: last_v = %s" % (epoch, last_v))
                        logging.warning("loss becomes nan during training!!!")
                        return 0
                    last_v = v

                # Evaluate if loss decreasing
                preds = []
                if epoch % EVALUATE_LOSS_AFTER == 0 or epoch == NEPOCH - 1:
                    loss = 0
                    r = 0
                    lr = -1
                    for step in range(BGN_LENGTH - PERIOD):
                        feed_dict = dict()
                        for i in range(PERIOD):  # 0 ~ 10
                            feed_dict[model.train_data[i]] = np.reshape(data[step + i], (data.shape[1], 1))
                        feed_dict[model.train_labels] = np.reshape(train_y[step + PERIOD], (1, 1))
                        _l, r, lr, predictions = session.run([model.loss, model.regulization, model.learning_rate, model.train_prediction], feed_dict=feed_dict)
                        preds.append(predictions)
                        loss += _l
                    loss = loss / (BGN_LENGTH - PERIOD)
                    logging.debug("%s epoch %s: loss = %s, regulization = %s, lr = %s" % (seed, epoch, loss, r, lr))
                    print("%s epoch %s: loss = %s, regulization = %s, lr = %s" % (seed, epoch, loss, r, lr))

            # save model parameters
            # model.saver.save(session, data_path)
            # print("Model saved in file: %s" % data_path)
            # preds = np.array(preds)
            # with open(pickle_path, 'wb') as f:
            #     pickle.dump(preds, f)
    return preds


def test_model(model, test_data, test_y, id=None):

    BGN_LENGTH = np.int(np.ceil(test_data.shape[0] * RATIO))

    with tf.Session(graph=model.graph) as session:

        preds = []

        tf.global_variables_initializer().run()

        times_predicted_same_direction = 0
        loss = 0
        for step in range(BGN_LENGTH - PERIOD):

            feed_dict = dict()
            for i in range(PERIOD):  # 0 ~ 10
                feed_dict[model.train_data[i]] = np.reshape(test_data[step + i], (test_data.shape[1], 1))
            feed_dict[model.train_labels] = np.reshape(test_y[step + PERIOD], (1, 1))
            _l, r, lr, predictions = session.run([model.loss, model.regulization, model.learning_rate, model.train_prediction], feed_dict=feed_dict)

            preds.append(predictions)
            loss += _l
            if (test_y[step + PERIOD] < 0 and predictions >= 0) or (test_y[step + PERIOD] >= 0 and predictions < 0):
                times_predicted_same_direction += 1

        loss = loss / (BGN_LENGTH - PERIOD)
        # direction_loss = (times_predicted_same_direction / (BGN_LENGTH - PERIOD))
        # logging.debug("direction loss = %s, average loss = %s, regulization = %s, lr = %s" % (direction_loss, loss, r, lr))
        # print("direction loss = %s, average loss = %s, regulization = %s, lr = %s" % (direction_loss, loss, r, lr))
        logging.debug("average loss = %s, regulization = %s, lr = %s" % (loss, r, lr))
        print("average loss = %s, regulization = %s, lr = %s" % (loss, r, lr))

        # save model parameters
        # model.saver.save(session, data_path)
        # print("Model saved in file: %s" % data_path)
        # preds = np.array(preds)
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump(preds, f)

    return preds


def get_para_name(prefix="model", seed=0):
    return str(RATIO) + '_' + str(seed)
    # return prefix + '_' + str(RATIO) + '_' + str(PERIOD) + '_' + str(NEPOCH) + '_' + str(
    #     UPDATE_NEPOCH) + '_' + str(LEARNING_RATE) + '_' + str(LAMDA) + '_' + str(NTOPICS) + '_' + str(seed)
