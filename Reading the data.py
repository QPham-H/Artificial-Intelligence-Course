import numpy as np
import sys

X = np.zeros((1,28*28))
#with open('trainingimages') as train_images:
#    for example in range(0,1):
#        for y in range(0,28):
#            for x in range(0,28):
#                i = train_images.read(1)
#                if i == '+':
#                    #X[example][28*y+x] = 0.5
#                    X[example,y,x] = 0.5
#                    sys.stdout.write('+')
#                elif i == '#':
#                    X[example,y,x] = 1
#                    sys.stdout.write('#')
#                elif i == ' ':
#                    sys.stdout.write(' ')
#            train_images.read(1)
#            sys.stdout.write('\n')

print('This is the picture\n')
#print(np.matrix(X))

print ('\n')
print ('\n') 

## Obtain Labels for Training
y_array = np.zeros((5000))
TrainingLab = open("traininglabels", "r")
Y = TrainingLab.readlines()
for x in range(0,len(Y)):
	y_array[x] = int(Y[x])
print(y_array)
TrainingLab.close()
