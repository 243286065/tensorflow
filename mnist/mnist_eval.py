#coding=utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py和mnist_train.py中定义的常量和函数
import mnist_inference
import mnist_train

# 每10秒加载一次最新的模型，并在测试数据集上测试最新的正确率
EVAL_INTERBAL_SEC = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 直接调用封装好的函数来计算正向传播结果
        # 测试时并不关心正则化损失的值，因此第二个参数设为None
        y = mnist_inference.inference(x, None)

        # 使用前向传播的结果计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        varibale_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        varibales_to_restore = varibale_averages.variables_to_restore()
        saver = tf.train.Saver(varibales_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时的迭代轮次
                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]
                    accuracy_score = sess.run(
                        accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (
                        global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
            time.sleep(EVAL_INTERBAL_SEC)


def main(argv=None):
    mnist = input_data.read_data_sets("data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
