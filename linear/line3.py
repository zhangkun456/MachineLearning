#!/usr/bin/python
# -*- coding:utf-8 -*-
# http://qianhk.com/2018/02/%E5%AE%A2%E6%88%B7%E7%AB%AF%E7%A0%81%E5%86%9C%E5%AD%A6%E4%B9%A0ML-%E7%94%A8TensorFlow%E5%AE%9E%E7%8E%B0%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%95/

#求 y=kx+3的k值

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

ops.reset_default_graph()

sess = tf.Session()

data_amount = 10 # 数据数量
batch_size = 25 # 批量大小

# 造数据 y=K * x * x + L (K=5,L=3)
x_vals = np.linspace(2, 10, data_amount)

y_vals = np.multiply(x_vals, x_vals)

y_vals = np.multiply(y_vals, 5)

y_vals = np.add(y_vals, 3)

# 生成一个N(0,15)的正态分布一维数组
y_offset_vals = np.random.normal(0, 15, data_amount)
y_vals = np.add(y_vals, y_offset_vals) # 为了有意使的y值有所偏差


print('y_vals=' + str(y_vals))

# 创建占位符
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 构造K 就是要训练得到的值
K = tf.Variable(tf.random_normal(mean=0, shape=[1, 1]))

calcY = tf.add(np.multiply(np.multiply(x_data, x_data), K), 3)

# print('calcY=' + str(calcY))

# 真实值与模型估算的差值
loss = tf.reduce_mean(tf.square(y_target - calcY))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.0000005)
train_step = my_opt.minimize(loss) # 目的就是使损失值最小

loss_vec = [] #保存每次迭代的损失值，为了图形化

KValue = 0

#print('x_data=' + str(x_vals))

for i in range(2000):
    rand_index = np.random.choice(data_amount, size=batch_size)
    x = np.transpose([x_vals[rand_index]])
    y = np.transpose([y_vals[rand_index]])
    val = sess.run(train_step, feed_dict={x_data: x, y_target: y})

    tmp_loss = sess.run(loss, feed_dict={x_data: x, y_target: y})
    loss_vec.append(tmp_loss)
    KValue = sess.run(K)
# 每25的倍数输出往控制台输出当前训练数据供查看进度
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' K = ' + str(KValue)+' tmp_loss='+str(tmp_loss))
        # print('Loss = ' + str(sess.run(loss, feed_dict={x_data: x, y_target: y})))

# 当训练完成后k的值就是当前的得到的结果，可以通过sess.run(K)取得
sess.close()





#展示结果
print('----  K = ' + str(KValue))

k = KValue[0]
#KValue = 5.008528
best_fit = []
for i in x_vals:
    best_fit.append(k * i * i + 3)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Base fit line')
#plt.plot(loss_vec, 'k-')
plt.title('Batch Look Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()