from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]= '0'

tempStore = os.path.join('../', 'tempData')

# set False if want to use pretrained model
overwrite = True

# this is the size of our encoded representations
encoding_dim = 8 # determined empirically

# this is our input placeholder;
input_pack = Input(shape=(39, ))

my_epochs = 5000
my_batch_size = 128
learningRate = 1e-3

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim * 4, activation='relu')(input_pack)
encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
decoded = Dense(39, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_pack, decoded)

# Separate Encoder model

# this model maps an input to its encoded representation
encoder = Model(input_pack, encoded)

# Separate Decoder model

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim, ))
# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# Train to reconstruct MNIST digits

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer=Adam(lr=learningRate), loss='mse')
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#---------------------------------#
# Loading data
#---------------------------------#

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30) 
pack_train = np.load(os.path.join(tempStore,'tcp_normal.npy'))
# data normalization
pack_train = (pack_train - np.mean(pack_train, axis=0)) / np.std(pack_train, axis=0)
pack_train[:,5] = 0.
pack_train[:,17] = 0.

# take a mini batch to testify the model
# mini_train = pack_train[0:500]

# normalize data before training
# pack_train = pack_train.astype('float32') / 255.

# load pretrained weights if nesessary
weightDir = os.path.join(tempStore, 'weights.h5')
if (not overwrite) and (os.path.exists(weightDir)):
    autoencoder.load_weights(weightDir)

model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

# Train autoencoder for 100 epochs
autoencoder.fit(pack_train, pack_train, epochs=my_epochs, batch_size=my_batch_size, shuffle=True,\
                validation_split=0.2, callbacks=[model_checkpoint, early_stop])
#---------------------------------#
# Evaluation
#---------------------------------#
pack_attack = np.load(os.path.join(tempStore,'tcp_attack.npy'))
pack_attack = (pack_attack - np.mean(pack_attack, axis=0)) / np.std(pack_attack, axis=0)
pack_attack[:,5] = 0.
pack_attack[:,17] = 0.

normal_test = np.load(os.path.join(tempStore,'tcp_normal_test.npy'))
normal_test = (normal_test - np.mean(normal_test, axis=0)) / np.std(normal_test, axis=0)
normal_test[:,5] = 0.
normal_test[:,17] = 0.

print(autoencoder.evaluate(pack_attack, pack_attack, batch_size=32))
print(autoencoder.evaluate(normal_test, normal_test, batch_size=32))


