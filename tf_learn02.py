import tensorflow as tf

### 计算图的使用

# a = tf.constant([1.0, 2.0], name='a')
# b = tf.constant([2.0, 3.0], name='b')
# result = a + b
# # 在编写程序过程中，TensorFlow会自动将定义的计算转化为计算图上的节点，
# # 在TensorFlow中，系统会自动维护一个默认的计算图，
# # 通过tf.get_default_graph函数可以获取当前默认的计算图。

# # 通过a.graph可以查看张量所属的计算图，因为没有特意指定，所以这个计算图应该是默认的计算图
# print(a.graph is tf.get_default_graph())

### 创建新的计算图
g1 = tf.Graph()
with g1.as_default():
    # 在g1中定义
    v = tf.get_variable(name="v", shape=[1], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    # 在g2中定义
    v = tf.get_variable(name="v", shape=[1], initializer=tf.ones_initializer())

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 在g1中，变量v取值应该为0，下面输出应该为[0.]
        print(sess.run(tf.get_variable('v')))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # 在g2中，变量v取值应该1，下面输出最应该为[1.0]
        print(sess.run(tf.get_variable("v")))


# with tf.variable_scope('v_scope') as scope1:
#     Weights1 = tf.get_variable('Weights', shape=[2,3])
#     bias1 = tf.get_variable([0.52], name='bias')

# print(Weights1.name)
# print(bias1.name)

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
# with tf.variable_scope('v_scope', reuse=True) as scope2:
#     Weights2 = tf.get_variable('Weights')
#     bias2 = tf.get_variable('bias', [1])  # ‘bias

# print(Weights2.name)
# print(bias2.name)