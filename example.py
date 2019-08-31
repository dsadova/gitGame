import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

data = pd.read_csv('gaussDataForNet.csv')

n = data.shape[0]
p = data.shape[1]

data = data.values

train_start = 0
train_end = int(np.floor(0.9*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
y_train = data_train[:, 1:]
X_train = data_train[:, 0]
y_test = data_test[:, 1:]
X_test = data_test[:, 0]
X = tf.placeholder(dtype=tf.float32, shape=[None, 512])
Y = tf.placeholder(dtype=tf.float32, shape=[None])
input = 512
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

W_hidden_1 = tf.Variable(weight_initializer([input, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

mse = tf.reduce_mean(tf.squared_difference(out, Y))
opt = tf.train.AdamOptimizer().minimize(mse)

net = tf.Session()

net.run(tf.global_variables_initializer())

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

X_test = [i.split(' ') for i in X_test]
count_y = 0
for i in X_test:
    count_x = 0
    for j in i:
        X_test[count_y][count_x] = float(j)
        count_x+=1
    count_y+=1
epochs = 10
batch_size = 256

for e in range(epochs):

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        batch_x = [z.split(' ') for z in batch_x]
        count_y = 0
        for z in batch_x:
            count_x = 0
            for j in z:
                batch_x[count_y][count_x] = float(j)
                count_x += 1
            count_y += 1
        batch_y = batch_y.reshape(256)
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})


        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            #plt.savefig(file_name)
            plt.pause(0.01)

print pred

y_test = y_test.reshape(1000)
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)