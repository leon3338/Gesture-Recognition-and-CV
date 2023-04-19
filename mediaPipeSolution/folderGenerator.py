import os
import numpy as np

DATA_PATH = os.path.join('Test_Data')
actions = np.array(['yes', 'no', 'milk', 'coffee', 'small', 'large'])
num_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
