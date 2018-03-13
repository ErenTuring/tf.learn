#  coding:utf8
### 计算图的简单示例


import tensorflow as tf

## 使用一个图，简单的分为以下两步:
## 1. 创建图结构(定义变量/计算节点)

#声明w1,w2两个变量,这里还通过seed参数设定随机种子,保证每次运行结果一致
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#x设置为一个占位符
x = tf.placeholder(tf.float32, shape=(None, 2), name='input')

#矩阵乘法操作
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


## 2. 创建会话，计算图

#创建一个会话
sess = tf.Session()

#初始化w1,w2
sess.run(w1.initializer)
sess.run(w2.initializer)

#使用sess计算前向传播输出值
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
print()
sess.close()