# Import libraries
import tensorflow as tf
import numpy as np
import collections
import os
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import scipy.io
from time import sleep

#Validate created model with a compltely different dataset
D=24
batch_size = 100
VALID_SET_SIZE=20000

#Importing dataset
dataset_valid=scipy.io.loadmat('2017-10-23_18-07-28_big_dataset_v1.mat')
dataset_effort_valid=dataset_valid.get('effort')


valid_target=[]
for i in range(0, VALID_SET_SIZE):
    valid_target.append(dataset_effort_valid[i,D])
valid_target=np.asarray(valid_target)
valid_target=np.reshape(valid_target,[len(valid_target),1])


valid_features=[]
for i in range(0, VALID_SET_SIZE):
    valid_features.append(dataset_effort_valid[i,0:D])
valid_features=np.asarray(valid_features)
valid_features=np.reshape(valid_features,[len(valid_features),D])


val_margin=0.1
avg_acc=0
validation_prediction=[]
valid_batch=int(len(valid_features)/batch_size)

#LOAD trained neural network
sess=tf.Session()
saver = tf.train.import_meta_graph('NN_tension_pred_saver-200000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Get saved graph
graph=tf.get_default_graph()

# Get placeholder variables

Xin=graph.get_tensor_by_name("Xin:0")
y_=graph.get_tensor_by_name("y_:0")

#Compute and plot data
target=[]
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([], [],'-k',label='prediction')
line2, = ax.plot([], [],'-r',label='real')

ax.legend()


for j in range (valid_batch):
    inp_data = valid_features[j*batch_size:(j+1)*batch_size,:]
    target=np.append(target,valid_target[j*batch_size:(j+1)*batch_size,:])
    pred_v = sess.run(y_,feed_dict={Xin: inp_data})
    validation_prediction=np.append(validation_prediction,pred_v)
    line1.set_ydata(validation_prediction)
    line1.set_xdata(range(len(validation_prediction)))
    line2.set_ydata(target)
    line2.set_xdata(range(len(target)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.05)

while True:
    plt.pause(0.05)




