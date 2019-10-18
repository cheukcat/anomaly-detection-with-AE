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
encoding_dim = 16 # determined empirically

def L2_loss(Truth, Prediction):
    return np.sum(np.square(Truth - Prediction), axis=1)


# this is our input placeholder;
input_pack = Input(shape=(39, ))

my_epochs = 5000
my_batch_size = 128
learningRate = 1e-3

# "encoded" is the encoded representation of the inputs
encoded = Dense(encoding_dim * 2, activation='relu')(input_pack)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
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
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

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
pack_normal = np.load(os.path.join(tempStore,'tcp_normal.npy'))
# split the normal data
pack_train = pack_normal[:-11038]
pack_test = pack_normal[-11038:]

# data normalization
# pack_train = (pack_train - np.mean(pack_train, axis=0)) / np.std(pack_train, axis=0)
pack_max = np.max(pack_train, axis=0)
pack_train = pack_train / pack_max
pack_train[:,5] = 0.
pack_train[:,17] = 0.

pack_test = pack_test / pack_max
pack_test[:,5] = 0.
pack_test[:,17] = 0.
np.save(os.path.join(tempStore,'normal_origin.npy'), pack_test)

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
# Prediction
#---------------------------------#
pack_attack = np.load(os.path.join(tempStore,'tcp_attack.npy'))
# pack_attack = (pack_attack - np.mean(pack_attack, axis=0)) / np.std(pack_attack, axis=0)
pack_attack = pack_attack / pack_max
# Important: The normalization methods of test packs need to be further study.
# It's impossible in the real case that we know the maximum of the attack packs, because packs are data flow.
# The data could only be normalized by the trained data.
pack_attack[:,5] = 0.
pack_attack[:,17] = 0.
np.save(os.path.join(tempStore,'attack_origin.npy'), pack_attack)

attack_recon = autoencoder.predict(pack_attack, verbose=1)
np.save(os.path.join(tempStore,'attack_recon.npy'), attack_recon)

normal_recon = autoencoder.predict(pack_test, verbose=1)
np.save(os.path.join(tempStore,'normal_recon.npy'), normal_recon)

ntrain_recon = autoencoder.predict(pack_train, verbose=1)

L2_attack = L2_loss(pack_attack, attack_recon)
L2_normal = L2_loss(pack_test, normal_recon)
L2_ntrain = L2_loss(pack_train, ntrain_recon)

attack_Statistics = {}
attack_Statistics['mean'] = np.mean(L2_attack)
attack_Statistics['std'] = np.std(L2_attack)
attack_Statistics['max'] = np.amax(L2_attack)
attack_Statistics['min'] = np.amin(L2_attack)
print('L2 loss of Attack Packs:', attack_Statistics)
normal_Statistics = {}
normal_Statistics['mean'] = np.mean(L2_normal)
normal_Statistics['std'] = np.std(L2_normal)
normal_Statistics['max'] = np.amax(L2_normal)
normal_Statistics['min'] = np.amin(L2_normal)
print('L2 Loss of Normal Packs:', normal_Statistics)
ntrain_Statistics = {}
ntrain_Statistics['mean'] = np.mean(L2_ntrain)
ntrain_Statistics['std'] = np.std(L2_ntrain)
ntrain_Statistics['max'] = np.amax(L2_ntrain)
ntrain_Statistics['min'] = np.amin(L2_ntrain)
print('L2 Loss of Trained Packs:', ntrain_Statistics)