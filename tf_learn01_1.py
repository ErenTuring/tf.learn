#  coding:utf8
# ## 计算图的简单示例

import tensorflow as tf
import numpy as np
# # 使用一个图，简单的分为以下两步:
# # 1. 创建图结构(定义变量/计算节点)

# 声明w1,w2两个变量,这里还通过seed参数设定随机种子,保证每次运行结果一致
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# x设置为一个占位符
x = tf.placeholder(tf.float32, shape=(None, 2), name='input')

# 矩阵乘法操作
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# # 2. 创建会话，计算图

# 创建一个会话
sess = tf.Session()

# 初始化w1,w2
sess.run(w1.initializer)
sess.run(w2.initializer)

class_num = 2
input_image = np.random.random([1, 3, 3, 2])

g1 = tf.Graph()
with g1.as_default():
    i_img = tf.placeholder(tf.float32, shape=(None, None, None, class_num), name="One_hot_img")
    o_img = tf.placeholder(tf.float32, shape=(None, None, None, class_num), name="one_channel_img")
    o_img = tf.arg_max(i_img, -1)

with tf.Session(graph=g1) as sess1:
    output_image = sess1.run(o_img, feed_dict={i_img: input_image})

print(np.array(output_image))

# 使用sess计算前向传播输出值
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
print()
sess.close()
