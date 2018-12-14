#!/usr/bin/python
#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 相关常数
INPUT_NODE = 784    # 输入层的节点数（28*28=784）
OUTPUT_NODE = 10    # 输出层节点数。因为要区分数字，即0~9，因此输出层节点数是10

# 神经网络的参数
HIDDEN_LAYER_NODE = 500  # 隐藏层节点数，这里只使用一个隐藏层，该层有500个节点
BATCH_SIZE = 100    # 一个训练batch中的训练数据个数。数字越小，训练过程越接近随机梯度下降；数字越大，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8    # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则项在损失函数中的系数
TRAINING_STEPS = 30000000  # 训练轮次
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率

# 辅助函数，给定神经网络的所有参数，计算神经网络的前向传播结果。
# 这里使用ReLU激活函数组成的三层全连接神经网络。
# avg_class是一个计算平均值的类。
def interence(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有使用滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        #计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average来计算得出变量的滑动平均值，然后再计算相应神经元的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, HIDDEN_LAYER_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[HIDDEN_LAYER_NODE]))
    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数下神经网络前向传播结果
    y = interence(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮次的变量。因为这个变量不需要用于计算滑动平均值，所以这里指定该变量为不可训练（trainable=False）
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始滑动平均类、给定训练轮数的变量可以加快训练早起变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果
    average_y = interence(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    # 因为标准答案是一个长度为10的数组，而sparse_softmax_cross_entropy_with_logits函数需要提供一个正确答案的数字，
    # 所以需要使用tf.argmax函数来得到正确答案对应得到类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算权重
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又要更新每一个参数的滑动平均值。为了一次完成多个操作，Tensorflow提供了tf.control_dependencies和tf.group两种机制
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络的前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 计算平均值，即为这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据，通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            # 没1000轮在验证数据集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))

                # 产生这一轮使用的一个batch的训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))

# 主程序入口
def main(argv=None):
    # 声明处理mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("data", one_hot=True)
    train(mnist)

# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
                