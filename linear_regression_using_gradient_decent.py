import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# build data
points_num = 100
vectors = []

# use "numpy" to randomly plot point_num point
for i in range(points_num):
	# random num
	x1 = np.random.normal(0.0, 0.66)
	# y = 0.1x + 0.2 + random num
	y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
	vectors.append([x1, y1])

# fetch x and y out
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# build linear regression module
bias = tf.Variable(tf.zeros([1]))
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# the module graded y
y = weight * x_data + bias

# loss function
# for every dimension of the tensor, calculate ((y - y_data) ^ 2)
loss = tf.reduce_mean(tf.square(y - y_data))

# use gradient decent optimizer to optimize our loss func
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

plt.plot(x_data, y_data, "*", label="Original Data")
plt.title("Linear Regression using Gradient Decent")
with tf.Session() as session:
	# init all variable
	init = tf.global_variables_initializer()
	session.run(init)
	for step in range(40):
		session.run(train)
		curr_weight = session.run(weight)
		curr_bias = session.run(bias)
		print("Step:", step, "Loss:", session.run(loss), "[Weight:", curr_weight, ", Bias:", curr_bias, "]")
		plt.plot(x_data, curr_weight * x_data + curr_bias, label=str(step) + "th step")
	# plt.legend()
	plt.show()
