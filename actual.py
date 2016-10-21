#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import numpy as np
import tensorflow as tf
import random
import scipy.misc
import matplotlib.pyplot as plt
from ale_python_interface import ALEInterface

n_input = 84 * 84 * 4
x = tf.placeholder(tf.float32, shape=(84,84,4))

# epsilon, state list, action list, reward list, replay capacity, we also need to keep the action (transition) number, maybe?
epsilon = 0.05
actionNumber = 0                                                                
replayCapacity = 1000000   
stateList = []
actionList = []
rewardList = []
processedList = []
isTerminalList = []
gamma = 0.99
learning_rate = 0.01

np.set_printoptions(threshold=np.nan, suppress=True)
#method for preprocessing frames
#return tf.concat(3, [[tf.image.resize_images(tf.convert_to_tensor((stateList[num])[np.newaxis]), 84, 84)], [tf.image.resize_images(tf.convert_to_tensor((stateList[num-1])[np.newaxis]), 84, 84)], [tf.image.resize_images(tf.convert_to_tensor((stateList[num-2])[np.newaxis]), 84, 84)], [tf.image.resize_images(tf.convert_to_tensor((stateList[num-3])[np.newaxis]), 84, 84)]])
def preProcess(num):
	if num > 2:
		return scipy.misc.imresize(np.concatenate((stateList[num], stateList[num-1], stateList[num-2], stateList[num-3]), axis=2), (84,84,4))
	elif num == 2:
		return scipy.misc.imresize(np.concatenate((stateList[0], stateList[1], stateList[2], stateList[2]), axis=2), (84,84,4))
        elif num == 1:                                                          
		return scipy.misc.imresize(np.concatenate((stateList[0], stateList[num], stateList[1], stateList[1]), axis=2), (84,84,4))
        else:                                                          
		return scipy.misc.imresize(np.concatenate((stateList[0], stateList[0], stateList[0], stateList[0]), axis=2), (84,84,4))

# Create some wrappers for simplicity                                           
def conv2d(x, W, b, strides=1):                                                 
    	# Conv2D wrapper, with bias and relu activation                             
    	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')    
    	x = tf.nn.bias_add(x, b)                                                    
    	return tf.nn.relu(x)  

# Create model                                                                  
def conv_net(x, weights, biases):                                      
    	# Reshape input picture                                                     
    	x = tf.reshape(x, shape=[-1, 84, 84, 4])                                    
    	# Convolution Layer                                                         
    	conv1 = conv2d(x, weights['wc1'], biases['bc1'], 4)                            
	# Convolution Layer                                                         
    	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 2)                        
        # Convolution Layer                                                         
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 1)
	# Fully connected layer                                                     
    	# Reshape conv3 output to fit fully connected layer input                   
    	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])      
    	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])                 
    	fc1 = tf.nn.relu(fc1)                                                       
    	# Output, class prediction                                                  
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])                 
    	print out
	return out 

#def writeScreenToFile(cnt):
#	x = ale.getScreenGrayscale()	
#	with file(str(cnt)+'rgb.txt', 'w') as outfile:
#		for i in range(x.shape[0]):
#			for j in range(x.shape[1]):
#				outfile.write('(')
#				for k in range(x.shape[2]):
#					outfile.write(str(x[i][j][k]) + ',')
#				outfile.write(')'+'  ')
#			outfile.write('\n')
#		#resize
#		process1 = tf.image.resize_images(tf.convert_to_tensor(x[np.newaxis]), 110, 84)
#		#testing concat 4
#		process2 = tf.concat(3, [process1, process1, process1, process1])
#		outfile.write(repr(process2))

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(sys.argv[1])

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
yy = tf.placeholder(tf.float32, [None, len(legal_actions)]) 

# initialize the convnet with random weights                                    
weights = {                                                                     
    # 8x8 conv, 4 input, 32 outputs                                             
    'wc1': tf.Variable(tf.random_normal([8, 8, 4, 32])),                        
    # 4x4 conv, 32 inputs, 64 outputs                                           
    'wc2': tf.Variable(tf.random_normal([4, 4, 32, 64])),                       
    # 3x3 conv, 64 inputs, 64 outputs                                           
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64])),                       
    # fully connected, 7*7*64 inputs, 512 outputs                               
    'wd1': tf.Variable(tf.random_normal([7*7*64, 512])),                        
    # 1024 inputs, 10 outputs (class prediction)                                
    'out': tf.Variable(tf.random_normal([512, len(legal_actions)]))                      
}                                                                               
                                                                                
biases = {                                                                      
    'bc1': tf.Variable(tf.random_normal([32])),                                 
    'bc2': tf.Variable(tf.random_normal([64])),                                 
    'bc3': tf.Variable(tf.random_normal([64])),                                 
    'bd1': tf.Variable(tf.random_normal([512])),                                
    'out': tf.Variable(tf.random_normal([len(legal_actions)]))                           
}  

# Construct model                                                               
pred = conv_net(x, weights, biases) 
squared_error = tf.reduce_mean(tf.square(tf.sub(yy, pred)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(squared_error)
argMax = tf.argmax(pred, 1)
yMax = tf.reduce_max(pred)

# Evaluate model                                                                
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(yy, 1))                    
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  

# Initializing the tf variables                                                    
init = tf.initialize_all_variables() 

# Launch the graph                                                              
with tf.Session() as sess:                                                      
	sess.run(init)                                                              
	# Begin learning code
	timeStep = 0
	for episode in xrange(100000):
		#initialize the beginning screen, preprocess
  		s1 = ale.getScreenGrayscale()
		if (timeStep < replayCapacity):
  			stateList.append(s1)
		else:
			stateList[timeStep % replayCapacity] = s1
		preprocess1 = preProcess(timeStep % replayCapacity)
                if (timeStep < replayCapacity):                                 
                	processedList.append(preprocess1)
		else:                                                           
                        processedList[timeStep % replayCapacity] = preprocess1 
		while True:
                	if (timeStep < replayCapacity):                 
                                isTerminalList.append(0)                
                        else:                                           
                                isTerminalList[timeStep % replayCapacity] = 0
			a = None
			#generate random number between 0 and 1, if smaller random, if larger follow the policy
			randNum = random.uniform(0, 1)
			if (randNum < epsilon):
				# actions are stored as indicies
				a = random.randrange(len(legal_actions))			
   			else:
				a = sess.run(argMax, feed_dict={x: processedList[timeStep % replayCapacity]}) 

			print a
   			# Apply an action and get the resulting reward
    			# print a
			r = ale.act(legal_actions[a])
			if (timeStep < replayCapacity):
				actionList.append(a)
				rewardList.append(r)
			else:
				actionList[timeStep % replayCapacity] = a
				rewardList[timeStep % replayCapacity] = r
			timeStep = timeStep + 1

			# New screen
			s2 = ale.getScreenGrayscale()    
			stateList.append(s2)
			pp = preProcess(timeStep % replayCapacity)			
			processedList.append(pp)
			print s2.shape			

			if ale.game_over():                                     
                                if (timeStep < replayCapacity):                 
                                        isTerminalList.append(1)                
                                else:                                           
                                        isTerminalList[timeStep % replayCapacity] = 1
                                ale.reset_game()                                
                                break   
			else:
                                if (timeStep < replayCapacity):                 
                                        isTerminalList.append(0)                
                                else:                                           
                                        isTerminalList[timeStep % replayCapacity] = 0	
			# Sample random minibatch of transitions (with minibatch size = 1)
			rand = random.randint(0, timeStep - 1)
                        # perform gradient descent step                 
                        # basically we need to modify 1 component of pred, which is adding a vector like [0,0,...,1,0,0,...]
                        # first, get the current (randomly selected) state feedforward prediction:
                        prediction = sess.run(pred, feed_dict={x: processedList[rand % replayCapacity]})
			# print gamma * sess.run(yMax, feed_dict={x: processedList[(rand + 1) % replayCapacity]})
			# use numpy methods to get the correct y vector for learning
			if isTerminalList[(rand + 1) % replayCapacity] == 1:
				prediction[0][actionList[rand % replayCapacity]] = rewardList[rand % replayCapacity]
			else:
				prediction[0][actionList[rand % replayCapacity]] = rewardList[rand % replayCapacity] + gamma * sess.run(yMax, feed_dict={x: processedList[(rand + 1) % replayCapacity]})
			# the key
			result = sess.run(optimizer, feed_dict={x: processedList[rand % replayCapacity], yy: prediction})

			# testing
            		loss, acc = sess.run([squared_error, accuracy], feed_dict={x: processedList[rand % replayCapacity], yy: prediction})   
            		print("Iter " + str(timeStep) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
			
