# keras-notes
Personal notes on using [keras](https://keras.io)

## Example Programs

### [keras1.py](examples/keras1.py)

The hello world of keras.  Learns a simple 4-input mathematical function using a NN with one hidden layer.

### [keras2.py](examples/keras2.py)

Another simple example which learns the [sklearn digits](http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html).

Note: gives a warning message.  [known issue](https://github.com/tensorflow/tensorflow/issues/8253)


## Backend

Theano or TensorFlow?  Edit `$HOME/.keras/keras.json`

```
"backend": "tensorflow"
```
or 
```
"backend": "theano"
```


