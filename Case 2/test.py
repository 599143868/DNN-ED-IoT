import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import time



model_path = 'yourpath/model_saved/'
data_path = 'yourpath/Dataset/Dataset.mat'
input_length = 1
target_length = 6      
lr = 1e-3

data = scipy.io.loadmat(data_path)
data = tf.cast(data['Dataset'],dtype=tf.float32)
with tf.Session() as sess:
    data = data.eval()

data_input,data_target = np.split(data,[1,],axis=1)
ss = StandardScaler()
std_data_input = ss.fit_transform(data_input)




x = tf.placeholder(tf.float32,[None,input_length])
y = tf.placeholder(tf.float32,[None,target_length])
is_train = tf.placeholder_with_default(False,(),'is_train')

w1 = tf.Variable(tf.truncated_normal(shape=(input_length,200),stddev=0.1))
b1 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w2 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b2 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w3 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b3 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w4 = tf.Variable(tf.truncated_normal(shape=(200,target_length),stddev=0.1))
b4 = tf.Variable(tf.truncated_normal(shape=(1,target_length),stddev=0.1))




hidden1 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(x,w1),b1),training=is_train))
hidden2 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden1,w2),b2),training=is_train))
hidden3 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden2,w3),b3),training=is_train))
y_pre = tf.nn.relu(tf.add(tf.matmul(hidden3,w4),b4))
mse = tf.reduce_mean(tf.square(y-y_pre))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.train.AdamOptimizer(lr).minimize(mse)
train_op = tf.group([train_op, update_ops])


saver = tf.train.Saver()
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    mse_,pre = sess.run([mse,y_pre],feed_dict={x:std_data_input,y:data_target,is_train:False})

    save_path = 'yourpath/DNN_output.mat'
    scipy.io.savemat(save_path,{'name':pre})

