from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

my_epochs = 5000
my_batch_size = 128
learningRate = 1e-3
overwrite = False

threshold = 0.6
#---------------------------------#
# Loading data
#---------------------------------#
tempStore = os.path.join('../', 'tempData')

def L2_map(Truth, Prediction):
    return np.square(Truth - Prediction)

normal_origin = np.load(os.path.join(tempStore,'normal_origin.npy'))
attack_origin = np.load(os.path.join(tempStore,'attack_origin.npy'))
normal_recon = np.load(os.path.join(tempStore,'normal_recon.npy'))
attack_recon = np.load(os.path.join(tempStore,'attack_recon.npy'))

L2_normal = L2_map(normal_origin, normal_recon)
L2_attack = L2_map(attack_origin, attack_recon)

x_train = np.concatenate((L2_normal[:-1000], L2_attack[:-1000]), axis=0)
y_train = np.ones((len(x_train), 1))
y_train[:(len(x_train) // 2)]  = 0
# One for attack, Zero for normal. 
input_pack = Input(shape=(39, ))
FC = Dense(64, activation='relu')(input_pack)
FC = Dense(64, activation='relu')(FC)
Output = Dense(1, activation='sigmoid')(FC)

classifier = Model(input_pack, Output)
classifier.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy', metrics=['accuracy'])

weightDir = os.path.join(tempStore, 'weights_c.h5')
if (not overwrite) and (os.path.exists(weightDir)):
    classifier.load_weights(weightDir)

model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

# Train autoencoder for 100 epochs
classifier.fit(x_train, y_train, epochs=my_epochs, batch_size=my_batch_size, shuffle=True,\
                validation_split=0.2, callbacks=[model_checkpoint, early_stop])

attack_test = L2_attack[-1000:]
normal_test = L2_normal[-1000:]

attack_pred = classifier.predict(attack_test)
normal_pred = classifier.predict(normal_test)

acc_attack = np.sum((attack_pred > threshold).astype(int)) / 1000
acc_normal = np.sum((normal_pred < threshold).astype(int)) / 1000

print('True Positive rate is:', acc_attack)
print('True Negative rate is:', acc_normal)