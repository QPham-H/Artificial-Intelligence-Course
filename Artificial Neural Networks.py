## ECE448/CS440 Artificial Inteligence

## Hw7

## Neural Networks, Perceptrons

## Marvin Alexander Hernandez & Quoc Pham



from array import array

import numpy as np



########################### PART 1 ###########################

	    ######### SINGLE-LAYER PERCEPTRON ##########



########### Initial Parameters. ###########

TrainPictures = 5000

TestPictures = 1000

lines = 28

chars = 28

alpha = 0.01						# learning rate

xo = -1 							# bias input

wo = np.zeros((10))					# 10 bias weight, 1 for each perceptron	

wo[:] = 20							# bias weights initialized to 20

wi = np.zeros((10,28,28))	 	# 10 array containig random weights for each perceptron

accuracy = 0.						# for computing accuracy



print("Obtaining images for Training")

########### Obtain images for Training ############

TrainingImgs = open("trainingimages", "r")

X = np.zeros((TrainPictures,lines,chars)) ## Dimention, lines, rows

for p in range(0,TrainPictures):

	for y in range(0,lines):

		for x in range(0,chars):

			i = TrainingImgs.readline(1)

			if i == '+':

				X[p][y][x] = 0.5

			elif i == '#':

				X[p][y][x] = 1.

			elif i == " ":

				X[p][y][x] = 0.

		TrainingImgs.readline(1)

		TrainingImgs.readline(1)	### REMOVE THIS LINE IF YOU USE WINDOWS, KEEP IF MAC

TrainingImgs.close()

########## Obtain Labels for Training #############

TrainingLab = open("traininglabels", "r")

Y = TrainingLab.readlines()

for x in range(0,len(Y)):

	Y[x] = int(Y[x])

TrainingLab.close()



print("Obtaining data for Testing")

############ Obtain images for Testing #############

TestImgs = open("testimages", "r")

X2 = np.zeros((TestPictures,lines,chars)) ## Dimention, lines, rows

for p in range(0,TestPictures):

	for y in range(0,lines):

		for x in range(0,chars):

			i = TestImgs.readline(1)

			if i == '+':

				X2[p][y][x] = 0.5

			elif i == '#':

				X2[p][y][x] = 1.

			elif i == " ":

				X2[p][y][x] = 0.

		TestImgs.readline(1)

		TestImgs.readline(1) ### REMOVE THIS LINE IF YOU USE WINDOWS, KEEP IF MAC

TestImgs.close()

############## Obtain Labels for Testing ###########

TestLab = open("testlabels", "r")

Y2 = TestLab.readlines()

for x in range(0,len(Y2)):

	Y2[x] = int(Y2[x])

TestLab.close()



print("Training")

############# Neural_Network Training #################

## Forward Training ##

for epoch in range(0,10):

	for pic in range(0,TrainPictures):	# 5000training examples

		for prctrn in range(0,10):		# training for each perceptron 

			Z = X[pic]*wi[prctrn]		# wights and inputs multiplied together

			zo = xo*wo[prctrn]			# bias input bais weigth multiplied

			SumZ = sum(sum(Z))+zo		# Sum over all z's

			if SumZ > 1:				# Classification rule

				TresholdZ = 1

			else:

				TresholdZ = 0

			if Y[pic] == prctrn:		# labeling perceptron

				label = 1				

			else:

				label = 0

			## Update Rule ##				

			err = label - TresholdZ		# error

			dwi = alpha*err*X[pic]		# gradient for weights of current perceptron	

			dwo = alpha*err*xo			# gradient for bias weight of current perceptron

			wi[prctrn] += dwi			# update weights of perceptron

			wo[prctrn] += dwo			# update bias weight

			#print "pic:", pic, "prctrn:", prctrn, "image:", Y[pic], " SumZ:", SumZ, " TresholdZ:", TresholdZ, "label:", label

		#print "\n"

## DONE TRAINING ##



print("Testing")

############## Neural_Network Testing ###############

## Forward Testing ##

for pic in range(0,TestPictures):		# 1000 testing examples

	for prctrn in range(0,10):			# looping trough each perceptron 

		Z = X2[pic]*wi[prctrn]			# wights and inputs multiplied together

		zo = xo*wo[prctrn]				# bias input bais weigth multiplied

		SumZ = sum(sum(Z))+zo			# Sum over all z's

		## Selecting the most confident perceptron

		if prctrn == 0:

			maxY = SumZ

			maxPerceptron = prctrn

		elif SumZ > maxY:

			maxY = SumZ

			maxPerceptron = prctrn

	if Y2[pic] == maxPerceptron:

		accuracy += 1.

print "Done Testing \nAccuracy:",(accuracy/1000)*100, '%'



## Highest confidency was 85.5% -> alpha = 0.01, zero initialization, 10 epochs

## Decaying is the worst with 12.2%
