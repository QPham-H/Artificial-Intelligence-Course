from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import numpy as np

# Grab the training data
x_train = np.zeros((5000,28*28))
with open('trainingimages') as train_images:
    for example in range(0,5000):
        for y in range(0,28):
            for x in range(0,28):
                i = train_images.read(1)
                if i == '+':
                    x_train[example,28*y+x] = 0.5
                elif i == '#':
                    x_train[example,28*y+x] = 1
            train_images.read(1)

# Grab the training labels
y_array = np.zeros((5000))
TrainingLab = open("traininglabels", "r")
Y = TrainingLab.readlines()
for x in range(0,len(Y)):
	y_array[x] = int(Y[x])
TrainingLab.close()

y_train = to_categorical(y_array, num_classes=10)

# Grab the test data
x_test = np.zeros((1000,28*28))
with open('testimages') as train_images:
    for example in range(0,1000):
        for y in range(0,28):
            for x in range(0,28):
                i = train_images.read(1)
                if i == '+':
                    x_test[example,28*y+x] = 0.5
                elif i == '#':
                    x_test[example,28*y+x] = 1
            train_images.read(1)

# Grab the test labels
y_array2 = np.zeros((5000))
TrainingLab = open("testlabels", "r")
Y = TrainingLab.readlines()
for x in range(0,len(Y)):
	y_array2[x] = int(Y[x])
TrainingLab.close()

            
y_test = to_categorical(y_array2, num_classes=10)

# Generate dummy data
#import numpy as np
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(130, activation='relu', input_dim=28*28))
model.add(Dropout(0.5))
model.add(Dense(130, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(130, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Try nesterov = False
# sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=250)
score = model.evaluate(x_test, y_test, batch_size=100)
