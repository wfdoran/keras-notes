from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras import backend as Kback
import numpy as np

# constants that can change
num_conv_features = 4
dropout_rate = .1                 # .1 => .975 on testing data,  .0 => .944 on testing data
num_epochs = 100

# Two vectors are the same if in every coordinate the values
# are both above .5 or both below .5
def same(a,b):
    for i in range(len(a)):
        az = a[i] > .5
        bz = b[i] > .5
        if az != bz:
            return False
    return True

# Test a mode by running the input X through and comparing the 
# output against the given y.
def test_model(model, X, y):
    guess = model.predict(X)
    n_right = 0
    n_wrong = 0
    for i in range(len(y)):
        if same(guess[i], y[i]):
            n_right += 1
        else:
            n_wrong += 1
    pct = n_right / (n_right + n_wrong)
    print("    %6d %6d : %8.4f" % (n_right, n_wrong, pct))
    
# load the digit data.    
digits = load_digits()
X = digits.data
y = digits.target

# save off 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=1)

# reshape as 8x8 images
num_rows = 8
num_cols = 8
if Kback.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, num_rows, num_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, num_rows, num_cols)
    shape = (1, num_rows, num_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], num_rows, num_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_cols, 1)
    shape = (num_rows, num_cols, 1)

# convert the output from 0,1,...,9 to a 10-long vector // one-hot column 
num_cats = 10
y_train2 = to_categorical(y_train, num_classes = num_cats)
y_test2 = to_categorical(y_test, num_classes = num_cats)

# pick sizes for the intermediate layers
n_inputs = len(X[0])
n_outputs = len(y_train2[0])
n_hidden1 = 2 * (n_inputs + n_outputs)
n_hidden2 = 3 * (n_inputs + n_outputs) // 2

# set up the neural net
model = Sequential()
model.add(Conv2D(num_conv_features, kernel_size=(3,3), activation='relu', input_shape=shape))
model.add(Flatten())
model.add(Dense(n_hidden1, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(n_hidden2, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(n_outputs, activation='softmax'))

# compile it
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# train the neural net
XX = np.array(X_train, dtype=float)
yy = np.array(y_train2, dtype=float)
model.fit(XX, yy, epochs=num_epochs, verbose=1)

# how does it do on the training data?
print("On training data:")
test_model(model, XX, yy)

# more importantly, how does it do on the testing data?
XXX = np.array(X_test, dtype=float)
yyy = np.array(y_test2, dtype=float)
print("On testing data:")
test_model(model, XXX, yyy)

