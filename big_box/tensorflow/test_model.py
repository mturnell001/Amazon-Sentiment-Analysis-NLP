import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential

# load vectors
print('Loading x...')
x_test_scaled = joblib.load('bin/x_test_scaled.sav')

# load labels
print('Loading y...')
y_test_cat = joblib.load('bin/y_test_cat.sav')

# load model
print('Loading model...')
model = tf.keras.models.load_model('models/keras_nn_8682.h5')

# make predictions
print('Making predictions...')
model_loss, model_accuracy = model.evaluate(x_test_scaled, y_test_cat, verbose=2)
print(f'Model: Neural Network\nLoss: {model_loss}\nAccuracy: {model_accuracy}')