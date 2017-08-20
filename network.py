"""Auxiliary functions for building, training and inference of a neural network"""
import tensorflow as tf
import numpy as np
from utils import weight_variable, bias_variable, build_feed, submit  # r2


class Network(object):
    """ Abstract wrapper class for TensorFlow models, all the _create methods are mere placeholders for the call in
        the build_graph method, and all shall be overridden in the child classes.
        """

    def __init__(self, batch_size, learning_rate, layers):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.layers = layers
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        pass

    def _create_variables(self):
        pass

    def _create_network(self):
        pass

    def _create_loss(self):
        pass

    def _create_optimizer(self):
        pass

    def _create_summaries(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


class Mercedez(Network):
    """ Class for defining the NN model for the mercedez challenge from kaggle"""

    def _create_placeholders(self):
        with tf.name_scope('Inputs'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob = tf.placeholder(tf.float32)

    def _create_variables(self):
        with tf.device('/cpu:0'):
            # Weights and bias variables for 2 FC layers + readout
            with tf.name_scope('Layer1'):
                self.w1 = weight_variable([567, self.layers[1]], name='Weights1')
                self.b1 = bias_variable([self.layers[1]], name='Bias1')

            with tf.name_scope('Layer2'):
                self.w2 = weight_variable([self.layers[1], self.layers[2]], name='Weights2')
                self.b2 = bias_variable([self.layers[2]], name='Bias2')

            with tf.name_scope('Readout'):
                self.w3 = weight_variable([self.layers[2], 1], name='Readout_Weights')
                self.b3 = bias_variable([1], 'Readout_Bias')

    def _create_network(self):
        with tf.device('/cpu:0'):
            # 2 fully-connected hidden layers with dropout between the 2nd hidden layer and the readout layer
            with tf.name_scope('Layer1'):
                self.h1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1, name='Relu1')

            with tf.name_scope('Layer2'):
                self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2) + self.b2, name='Relu2')

            with tf.name_scope('Dropout'):
                self.h2_drop = tf.nn.dropout(self.h2, self.keep_prob, name='Dropout')

            with tf.name_scope('Readout'):
                self.y_fc = tf.matmul(self.h2_drop, self.w3) + self.b3

    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Loss'):
                # MSE loss since we're doing regression, other option would be using negative R squared as our
                # loss, since the model will be evaluated based on it
                self.loss = tf.losses.mean_squared_error(labels=self.y_, predictions=self.y_fc)

    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            # Adam optimizer as usual
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                     global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            tf.summary.scalar('Loss', self.loss)
            tf.summary.histogram('Histogram Loss', self.loss)

            self.summary_op = tf.summary.merge_all()

    def metrics(self, feed_dict, targets):
        """R2 and MSE metrics"""

        mse = self.loss.eval(feed_dict=feed_dict)
        r_squared = 1 - mse / np.var(targets)

        return mse, r_squared


def validate_train(session, fold, max_iter, dropout_rate, model, train_input, train_target, cv_input, cv_target):
    saver = tf.train.Saver()

    with session.as_default() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting Training...')

        # Batch training: building the feed_dicts does not have to be done inside the training loop
        train_feed = build_feed(model, train_input, train_target, dropout_rate)
        train_eval_feed = build_feed(model, train_input, train_target, 1.0)
        cv_eval_feed = build_feed(model, cv_input, cv_target, 1.0)

        for i in range(max_iter):
            # Logging loss to prompt every so often
            if i % 100 == 0:
                train_loss = model.loss.eval(feed_dict=train_eval_feed)
                cv_loss = model.loss.eval(feed_dict=cv_eval_feed)
                print('Step {0}, Train Loss: {1: .2f} | CV Loss: {2: .2f}'.format(i, train_loss, cv_loss))

            # Train Step
            model.train_op.run(feed_dict=train_feed)

        # Short training time, thus checkpoint is done by the end of training and not every N steps
        saver.save(sess, './checkpoints/fold', fold)

        # Log final model training and CV metrics
        (train_mse, train_r2) = model.metrics(train_eval_feed, train_target)
        (cv_mse, cv_r2) = model.metrics(cv_eval_feed, cv_target)
        print('\nTraining Finished, Training Loss: {0: .2f} | R_squared: {1: .5f}'.format(train_mse, train_r2))
        print('                 Validation Loss: {0: .2f} | R_squared: {1: .5f}\n'.format(cv_mse, cv_r2))

        return cv_mse, cv_r2


def infer_train(max_iter, dropout_rate, model, train_input, train_target, test_input, id_column):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting Training...')

        # Batch training: building the feed_dicts does not have to be done inside the training loop
        train_feed = build_feed(model, train_input, train_target, dropout_rate)
        train_eval_feed = build_feed(model, train_input, train_target, 1.0)

        for i in range(max_iter):
            # Logging loss to prompt every so often
            if i % 100 == 0:
                train_loss = model.loss.eval(feed_dict=train_eval_feed)
                print('Step {0}, Train Loss: {1: .2f}'.format(i, train_loss))

            # Train Step
            model.train_op.run(feed_dict=train_feed)

        # Short training time, thus checkpoint is done by the end of training and not every N steps
        saver.save(sess, './checkpoints/fold')

        # Log final model training and CV metrics
        (train_mse, train_r2) = model.metrics(train_eval_feed, train_target)
        print('\nTraining Finished, Training Loss: {0: .2f} | R_squared: {1: .5f}'.format(train_mse, train_r2))

        inference = model.y_fc.eval(feed_dict={model.x: test_input, model.keep_prob: 1.0})
        submit(id_column, inference)
