import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard





num_sequences = 30
sequence_length = 30
DATA_PATH = os.path.join('CVGR_Data')
actions = np.array(['background', 'yes', 'no', 'milk', 'coffee', 'small', 'large'])

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(num_sequences):
        window=[]
    for frame_num in range(sequence_length):
        res = np.load(os.path.join(DATA_PATH, action, str(sequence),"{}.npy".format(frame_num)))
        window.append(res)
    sequences.append(window)
    labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.05)

print(x_train.shape)
#print(np.array(sequences).shape)

#--------------------------------------------------------------------------

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=1000, callbacks=[tb_callback])
