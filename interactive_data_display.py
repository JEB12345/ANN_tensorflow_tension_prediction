# Import libraries
import tensorflow as tf
import numpy as np
import collections
import os
import collections
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import scipy.io
from time import sleep

# ROS Stuff
import rospy
from sensor_msgs.msg import JointState

#Define variables
D=24                                    #Number of features (in this case, number of motors)
batch_size = 100                        #Size of the batch to be fed in the NN
SET_SIZE=60*batch_size                  #Size of the dataset

# Intialze a ROS node
rospy.init_node('ANN_tension_predicition_v3')

# Create two buffers to "ping-pong" between
effort1 = np.ones([100,25])*np.nan
effort2 = np.ones([100,25])*np.nan
effort = [effort1, effort2]		# one array for both buffers addressable by arraySwitch defined below
index = 0 				# used to count buffer batch size
arraySwitch = 0 			# flag to change buffers
array1Full = 0				# flag to signal that buffer is full
array2Full = 0				# flag to signal that buffer is full

# ROS message callback for Hebi JointState message
def callback(msg):
    global effort			# global for scoping the variables outside of the callback function
    global index
    global arraySwitch
    global array1Full
    global array2Full

# Switches between the two buffers. When one fills up to the batch size, the other then start to fill up
# Probably could have used a switch statement, but this works
    if arraySwitch == 0:
        effort[arraySwitch][index,:] = msg.effort
        index += 1
        if index >= 100:
            arraySwitch = 1
            index = 0
            array1Full = 1
    elif arraySwitch == 1:
        effort[arraySwitch][index,:] = msg.effort
        index += 1
        if index >= 100:
            arraySwitch = 0
            index = 0
            array2Full = 1
    else:
        arraySwitch = 0
        index = 0

# Subscribes the the ROS message which houses the Hebi information
rospy.Subscriber("/hebiros/my_group/feedback/joint_state", JointState, callback)

#LOAD trained neural network
sess=tf.Session()
saver = tf.train.import_meta_graph('NN_tension_pred_saver-200000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

# Get saved graph
graph=tf.get_default_graph()

# Get placeholder variables
Xin=graph.get_tensor_by_name("Xin:0")
y_=graph.get_tensor_by_name("y_:0")


### Compute and plot data ###
target=[]
validation_prediction=[]

#Make plot interactive
plt.ion()

#Define figures and two lines that will be used to plot predicted and real data in the same plot
fig = plt.figure()
ax = fig.add_subplot(111)
pred_line, = ax.plot([], [],'-k',label='prediction')
real_line, = ax.plot([], [],'-r',label='real')

ax.legend()

#Feed batches of features to the NN and update plot
while not rospy.is_shutdown():
    if (array1Full==1 or array2Full==1):
        if array1Full==1:
            #get input data from buffer
            inp_data = effort[0][:,0:24]
            #append tension sensor data for plotting 
            target=np.append(target,effort[0][:,24])
            array1Full = 0
        if array2Full==1:
            #get input data from buffer
            inp_data = effort[1][:,0:24]
            #append tension sensor data for plotting 
            target=np.append(target,effort[1][:,24])
            array2Full = 0
        #Run the NN prediction
        pred_v = sess.run(y_,feed_dict={Xin: inp_data})
        #Append predicted value to the validation_prediction variable (used for the plot)
        validation_prediction=np.append(validation_prediction,pred_v)
        
        if (np.size(target) > SET_SIZE):
            np.delete(target,0)
        if (np.size(validation_prediction) > SET_SIZE):
            np.delete(validation_prediction,0)
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
        plt.pause(0.01)
        print('test')


