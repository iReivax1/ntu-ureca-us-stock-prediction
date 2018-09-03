import numpy as np
import tensorflow as tf


batch_size = 1

# Training Parameters
start_learning_rate = 0.005
training_steps = 10000
display_step = 10

# Network Parameters
num_input = 3       # stock price input
timesteps = 4       # consider ? time step
num_hidden = 64     # hidden layer num of features
num_classes = 3     # price fall, price no change, price rise

"""
how to determine if stock as at a given date is fall, nochange, rise
1. compute the normalize stock price over number of timesteps
2. price "far above" 0 mean is rise, near mean is nochange, below mean is fall
   for each timestep
"""


class LSTM():

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            # tf Graph input
            with g.name_scope("input_gate"):
                # Input gate: input, previous output, and bias.
                ix = tf.Variable(tf.truncated_normal([num_input, num_hidden], -0.1, 0.1))
                im = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], -0.1, 0.1))
                ib = tf.Variable(tf.zeros([1, num_hidden]))

            with g.name_scope("forget_gate"):
                # Forget gate: input, previous output, and bias.
                fx = tf.Variable(tf.truncated_normal([num_input, num_hidden], -0.1, 0.1))
                fm = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], -0.1, 0.1))
                fb = tf.Variable(tf.zeros([1, num_hidden]))

            with g.name_scope("memory_cell"):
                # Memory cell: input, state and bias.
                cx = tf.Variable(tf.truncated_normal([num_input, num_hidden], -0.1, 0.1))
                cm = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], -0.1, 0.1))
                cb = tf.Variable(tf.zeros([1, num_hidden]))

            with g.name_scope("output_gate"):
                # Output gate: input, previous output, and bias.
                ox = tf.Variable(tf.truncated_normal([num_input, num_hidden], -0.1, 0.1))
                om = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], -0.1, 0.1))
                ob = tf.Variable(tf.zeros([1, num_hidden]))

            saved_output = tf.Variable(tf.zeros([batch_size, num_hidden]), trainable=False)
            saved_state = tf.Variable(tf.zeros([batch_size, num_hidden]), trainable=False)

            # Classifier weights and biases.
            w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], -0.1, 0.1), name="weights")
            b = tf.Variable(tf.zeros([num_classes]), name="biases")

            def lstm_cell(i, o, state):
                input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
                forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
                update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
                state = forget_gate * state + input_gate * tf.tanh(update)
                output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
                return output_gate * tf.tanh(state), state

            # Input data.
            self.train_data = list()
            for _ in range(timesteps + 1):
                self.train_data.append(tf.placeholder(tf.float32, shape=[batch_size, num_input]))

            # output data.
            self.train_output = list()
            for _ in range(timesteps):
                self.train_output.append(tf.placeholder(tf.float32, shape=[batch_size, num_classes]))

            train_inputs = self.train_data[:]
            train_labels = self.train_output[:]

            # Unrolled LSTM loop.
            outputs = list()
            output = saved_output
            state = saved_state
            for i in train_inputs[:-1]:
                output, state = lstm_cell(i, output, state)
                outputs.append(output)

            with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
                # Classifier.
                logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.concat(train_labels, 0),
                        logits=logits
                    )
                )

            # Optimizer.
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.95, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

            # Predictions.
            train_prediction = tf.nn.softmax(logits)

            self.optimizer = optimizer
            self.loss = loss
            self.train_prediction = train_prediction
            self.learning_rate = learning_rate

    # def start_train(self, data, label):
    def start_train(self):
        with tf.Session(graph=self.graph) as sess:

            sess.run(tf.global_variables_initializer())

            optimizer = self.optimizer
            loss = self.loss
            train_prediction = self.train_prediction
            learning_rate = self.learning_rate

            data = [[1, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1],
                    [3, 1, 1],
                    [1, 1, 1]]

            label = [[1, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]

            mean_loss = 0
            for step in range(training_steps):
                feed_dict = dict()
                for i in range(timesteps + 1):
                    feed_dict[self.train_data[i]] = np.reshape(data[i], (batch_size, num_input))
                for i in range(timesteps):
                    feed_dict[self.train_output[i]] = np.reshape(label[i], (batch_size, num_classes))

                _, l, predictions, lr = sess.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
                mean_loss += l

                if step % display_step == 0:
                    if step > 0:
                        mean_loss = mean_loss / display_step
                        print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                        print(predictions)
                        mean_loss = 0
