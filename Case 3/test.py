import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import time


#参数设置
model_path = 'C:/Users/Administrator.JACL4WGNSVYXUQ6/Desktop/Fourth Paper/材料/代码/dynamic units status/model_saved/'
data_path = 'C:/Users/Administrator.JACL4WGNSVYXUQ6/Desktop/Fourth Paper/材料/代码/dynamic units status/Dataset/Dataset.mat'
input_length = 2
target_length = 6      #分解出的系数的个数
lr = 1e-3
Alpha = [0.00375,0.0175,0.0625,0.00834,0.025,0.025]
Beta = [2,1.75,1.0,3.25,3.0,3.0]
#数据读取与预处理
data = scipy.io.loadmat(data_path)
data = tf.cast(data['Dataset'],dtype=tf.float32)
with tf.Session() as sess:
    data = data.eval()
# np.random.shuffle(data)
data_input,data_target = np.split(data,[2,],axis=1)
ss = StandardScaler()
std_data_input = ss.fit_transform(data_input)

#占位定义
x = tf.placeholder(tf.float32,[None,input_length])
y = tf.placeholder(tf.float32,[None,target_length])
is_train = tf.placeholder_with_default(False,(),'is_train')
#隐藏层定义
w1 = tf.Variable(tf.truncated_normal(shape=(input_length,200),stddev=0.1))
b1 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w2 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b2 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w3 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b3 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w4 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b4 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w5 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b5 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w6 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b6 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w7 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b7 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w8 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b8 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w9 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b9 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w10 = tf.Variable(tf.truncated_normal(shape=(200,200),stddev=0.1))
b10 = tf.Variable(tf.truncated_normal(shape=(1,200),stddev=0.1))
w11 = tf.Variable(tf.truncated_normal(shape=(200,target_length),stddev=0.1))
b11 = tf.Variable(tf.truncated_normal(shape=(1,target_length),stddev=0.1))

hidden1 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(x,w1),b1),training=is_train))
hidden2 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden1,w2),b2),training=is_train))
hidden3 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden2,w3),b3),training=is_train))
hidden4 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden3,w4),b4),training=is_train))
hidden5 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden4,w5),b5),training=is_train))
hidden6 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden5,w6),b6),training=is_train))
hidden7 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden6,w7),b7),training=is_train))
hidden8 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden7,w8),b8),training=is_train))
hidden9 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden8,w9),b9),training=is_train))
hidden10 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(hidden9,w10),b10),training=is_train))
y_pre = tf.nn.relu(tf.add(tf.matmul(hidden10,w11),b11))
msee = tf.reduce_mean(tf.square(y-y_pre),axis=1)
mse = tf.reduce_mean(tf.square(y-y_pre))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.train.AdamOptimizer(lr).minimize(mse)
train_op = tf.group([train_op, update_ops])
mse_train_best = 5000000

#开始测试
saver = tf.train.Saver()
with tf.Session() as sess:

    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    mse_,pre = sess.run([msee,y_pre],feed_dict={x:std_data_input,y:data_target,is_train:False})

    DNN_cost = [(Alpha[0]*x[0]**2+Beta[0]*x[0]+
                 Alpha[1]*x[1]**2+Beta[1]*x[1]+
                 Alpha[2]*x[2]**2+Beta[2]*x[2]+
                 Alpha[3]*x[3]**2+Beta[3]*x[3]+
                 Alpha[4]*x[4]**2+Beta[4]*x[4]+
                 Alpha[5]*x[5]**2+Beta[5]*x[5]) for x in pre]

    Lambda_cost = [(Alpha[0]*x[0]**2+Beta[0]*x[0]+
                    Alpha[1]*x[1]**2+Beta[1]*x[1]+
                    Alpha[2]*x[2]**2+Beta[2]*x[2]+
                    Alpha[3]*x[3]**2+Beta[3]*x[3]+
                    Alpha[4]*x[4]**2+Beta[4]*x[4]+
                    Alpha[5]*x[5]**2+Beta[5]*x[5]) for x in data_target]

    fig = plt.figure(1,figsize=(12, 10))

    ax = fig.add_subplot(111,projection='3d')
    print(np.mean(np.array(DNN_cost)-np.array(Lambda_cost))/np.mean(np.array(Lambda_cost)))
    cm = plt.cm.get_cmap('viridis')
    sc =  ax.scatter3D(data_input[:,0],data_input[:,1],Lambda_cost,c = np.array(DNN_cost)-np.array(Lambda_cost),cmap = cm,vmin=0)
    ax.set_xlabel(r'$P_d$',fontdict={'size':15})
    ax.set_ylabel(r'$U_{1\sim N}$',fontdict={'size':15})
    ax.set_zlabel(r'Total Cost (DNN)',fontdict={'size':15})
    ax.view_init(elev=20,azim=169)
    fig.subplots_adjust(right=0.8)
    # colorbar 左 下 宽 高
    l = 0.76
    b = 0.20
    w = 0.015
    h = 0.48

    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)

    cp = plt.colorbar(sc, cax=cbar_ax)
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    cp.set_label('Difference in Cost (DNN minus $\lambda$-ITE)', fontdict={'size':15})
    # plt.savefig('case1.pdf', bbox_inches = 'tight',)
    plt.show()




    fig = plt.figure(2,figsize=(12, 10))

    ax = fig.add_subplot(111,projection='3d')

    cm = plt.cm.get_cmap('viridis')
    sc =  ax.scatter3D(data_input[:,0],data_input[:,1],data_target[:,2],c = mse_,cmap = cm,vmin=0)
    ax.set_xlabel(r'$P_d$',fontdict={'size':15})
    ax.set_ylabel(r'$U_{1\sim N}$',fontdict={'size':15})
    ax.set_zlabel(r'$P_3$',fontdict={'size':15})
    ax.view_init(elev=20,azim=169)
    fig.subplots_adjust(right=0.8)
    # colorbar 左 下 宽 高      l = 0.82
    #     b = 0.15
    #     w = 0.015
    #     h = 0.65
    l = 0.75
    b = 0.20
    w = 0.015
    h = 0.48

    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)

    cp = plt.colorbar(sc, cax=cbar_ax)
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    cp.set_label('Mean Square Error', fontdict={'size':15})
    # plt.savefig('case1.pdf', bbox_inches = 'tight',)
    plt.show()