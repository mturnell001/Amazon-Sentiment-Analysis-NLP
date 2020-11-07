# imports
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load vectors
print('Loading x...')
x_train_scaled = joblib.load('bin/x_train_scaled.sav')
x_test_scaled = joblib.load('bin/x_test_scaled.sav')

# load labels
print('Loading y...')
y_train_cat = joblib.load('bin/y_train_cat.sav')
y_test_cat = joblib.load('bin/y_test_cat.sav')

# create model
print('Creating model...')
num_hidden_nodes = 100
input_dim = x_train_scaled.shape[1]
model = Sequential()
model.add(Dense(units=num_hidden_nodes, activation='relu', input_dim=input_dim))
model.add(Dense(units=num_hidden_nodes, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))
model.summary()

# compile model
print('Compiling model...')
model.compile(optimizer='adam', oss='binary_crossentropy', metrics=['accuracy'])

# requires 48 < x <= 64 GB of system memory
# took 12 hours or approximately 429 seconds per epoch with 12 threads
# train model
print('Training model...')
model.fit(x_train_scaled, y_train_cat, epochs=100, shuffle=True, verbose=2)

print('Saving model...')
model.save('bin/keras_nn.h5')

# make predictions
print('Making predictions...')
model_loss, model_accuracy = model.evaluate(x_test_scaled, y_test_cat, verbose=2)
print(f'Model: Neural Network\nLoss: {model_loss}\nAccuracy: {model_accuracy}')