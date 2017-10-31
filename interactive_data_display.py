# Import libraries
import tensorflow as tf
import numpy as np
import collections
import os
import collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import scipy.io
from time import sleep

#Define variables
D=24                                    #Number of features (in this case, number of motors)
batch_size = 100                        #Size of the batch to be fed in the NN
VALID_SET_SIZE=20000                    #Size of the dataset

#Importing dataset
dataset_valid=scipy.io.loadmat('2017-10-23_18-07-28_big_dataset_v1.mat')
dataset_effort_valid=dataset_valid.get('effort')

#Importing target(output) values that will be plotted next to the prediction
valid_target=[]
for i in range(0, VALID_SET_SIZE):
    valid_target.append(dataset_effort_valid[i,D])
valid_target=np.asarray(valid_target)
valid_target=np.reshape(valid_target,[len(valid_target),1])

#Importing features(input) values that will be fed to the NN
valid_features=[]
for i in range(0, VALID_SET_SIZE):
    valid_features.append(dataset_effort_valid[i,0:D])
valid_features=np.asarray(valid_features)
valid_features=np.reshape(valid_features,[len(valid_features),D])



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

#Make plot interactive
plt.ion()

#Define figures and two lines that will be used to plot predicted and real data in the same plot
fig = plt.figure()
ax = fig.add_subplot(111)
pred_line, = ax.plot([], [],'-k',label='prediction')
real_line, = ax.plot([], [],'-r',label='real')

ax.legend()

#Feed batches of features to the NN and update plot
for j in range (valid_batch):
    #select input data from dataset
    inp_data = valid_features[j*batch_size:(j+1)*batch_size,:]
    #append real output data to target variable (used to plot real data)
    target=np.append(target,valid_target[j*batch_size:(j+1)*batch_size,:])
    #Run the NN prediction
    pred_v = sess.run(y_,feed_dict={Xin: inp_data})
    #Append predicted value to the validation_prediction variable (used for the plot)
    validation_prediction=np.append(validation_prediction,pred_v)
    #Define X and Y axis variable for the two lines in the plot
    pred_line.set_ydata(validation_prediction)
    pred_line.set_xdata(range(len(validation_prediction)))
    real_line.set_ydata(target)
    real_line.set_xdata(range(len(target)))
    #Scale plot
    ax.relim()
    ax.autoscale_view()
    #Update
    plt.draw()
    plt.pause(0.05)

while True:
    plt.pause(0.05)




