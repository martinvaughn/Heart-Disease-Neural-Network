import numpy as np
import pandas as pd 
import sklearn.utils
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Read in csv filename
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = os.path.join(dirname, filename)
        
df = pd.read_csv(file) # create data frame from heart_disease csv

df = sklearn.utils.shuffle(df) # Shuffle data set
df = df.reset_index(drop=True)

a = pd.get_dummies(df['cp'], prefix="cp") # create dummie variables for chest pains
frames = [df, a]
df = pd.concat(frames, axis=1)

# only keeps inputs for AGE, SEX, ChestPains, Resting blood pressure, max_heart_rate, exercise_induced_pain & diagnoses
df = df.drop(columns=['cp', 'thal', 'slope', 'chol', 'fbs', 'restecg', 'oldpeak', 'ca'])

training = df[:245] # split training & testing data 0-245 
testing = df[246:] # 246-297

scaler = MinMaxScaler(feature_range=(0, 1)) #scale all data to be in range [0,1)
scaled_training = scaler.fit_transform(training) 
scaled_testing = scaler.transform(testing)

scaled_training_df = pd.DataFrame(scaled_training, columns=training.columns.values) 
scaled_testing_df = pd.DataFrame(scaled_testing, columns=testing.columns.values)

training_df = scaled_training_df
testing_df = scaled_testing_df

# separate x & y values
y_train = training_df.condition.values # create y data from the diagnoses (condition) column alone
x_train = training_df.drop(['condition'], axis=1) # create x data from all except condition column

y_test = testing_df.condition.values
x_test = testing_df.drop(['condition'], axis=1)


# COLUMNS = ['age', 'sex', 'trestbps', 'thalach', 'exang', 'cp_0', 'cp_1', 'cp_2', 'cp_3']

#create the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu')) # 50 best
model.add(Dense(100, activation='relu'))  # 100 best
model.add(Dense(50, activation='relu')) # 50 best

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, shuffle=True, batch_size=300, validation_data=(x_test, y_test)) #300 batch_best

# Test the model
metrics = model.evaluate(x_test, y_test)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))

model.save('v6_heart_model.h5')

#convert to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_buffer = converter.convert()

open file and write to it
open('v6_heart_model.tflite', 'wb').write(tflite_buffer)
print("TfLite model created")
