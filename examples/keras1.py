from keras.models import Sequential
from keras.layers.core import Dense,Activation
import random
import numpy as np

def test_model(model, X, y):
    guess = model.predict(X)
    count = [[0,0],[0,0]]
    for i in range(len(y)):
        count[y[i][0] > .5][guess[i][0] > .5] += 1
    n_right = count[0][0] + count[1][1]
    n_wrong = count[1][0] + count[0][1]
    pct = n_right / (n_right + n_wrong)
    print("    %6d %6d %6d %6d : %8.4f" % (count[0][0], count[0][1], count[1][0], count[1][1], pct))

# A classic neural net with 4 inputs, 2 output, and one hidden layer
# with 6 nodes using the tanh function 
n_inputs = 4
n_hidden = 6
n_outputs = 2

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_inputs))
model.add(Activation('relu'))
model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dense(n_outputs))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')

# We want to see if the neural net can learn the boolean function
#
#   x0 * x1 + x1 * x2 + x3
#
# from slightly noisy data

def and_func(a,b):
    return a * b
   
def xor_func(a,b):
    return a + b - 2 * a * b
    
def f(x):
    t1 = and_func(x[0], x[1])
    t2 = and_func(x[1], x[2])
    t3 = xor_func(t1,t2)
    t4 = xor_func(t3, x[3])
    t5 = t4 + random.random() / 100.0
    if t5 < 0.0:
        t5 = 0.0
    if t5 > 1.0:
        t5 = 1.0
    return 1.0 if t5 > .5 else 0.0
    
# training data
n_train = 10000
in_data = [[random.random() for i in range(n_inputs)] for j in range(n_train)]
out_data = [[f(x), 1-f(x)] for x in in_data]

X_train = np.array(in_data)
y_train = np.array(out_data)

# train on the training data
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
print("On training data:")
test_model(model, X_train, y_train)

# test on test data
n_test = 100
in_data = [[random.random() for i in range(n_inputs)] for j in range(n_test)]
out_data = [[f(x), 1-f(x)] for x in in_data]

X_test = np.array(in_data)
y_test = np.array(out_data)
print("On test data:")
test_model(model, X_test, y_test)

# test on some random data
in_data = [[random.random() for i in range(n_inputs)] for j in range(n_test)]
bogus = [1.0 if random.random() > .5 else 0.0 for j in range(n_test)]
out_data = [[b, 1-b] for b in bogus]

X_test = np.array(in_data)
y_test = np.array(out_data)
print("On random data:")
test_model(model, X_test, y_test)

