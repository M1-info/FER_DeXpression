from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, concatenate, Flatten, Dense, Dropout, Input, Activation
import numpy as np

features = np.load('./deXpression_features.npy')
labels = np.load('./deXpression_labels.npy')

imagesize = 48
dropout_rate = 0.8
padding = 'valid'

# Input layer
inputs = Input(shape=(imagesize, imagesize, 1))

x = Conv2D(64, (7, 7), strides=(2, 2), padding=padding)(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding)(x)
x = BatchNormalization()(x)

net_1 = Conv2D(96, (1, 1), padding=padding)(x)
net_1 = Activation('relu')(net_1)
net_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding=padding)(x)
net_3 = Conv2D(208, (3, 3), padding=padding)(net_1)
net_3 = Activation('relu')(net_3)
net_4 = Conv2D(64, (1, 1), padding=padding)(net_2)
net_4 = Activation('relu')(net_4)
chunk_1 = concatenate([net_3, net_4], axis=3)

net_5 = Conv2D(96, (1, 1), padding=padding)(chunk_1)
net_5 = Activation('relu')(net_5)
net_6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding=padding)(chunk_1)
net_7 = Conv2D(208, (3, 3), padding=padding)(net_5)
net_7 = Activation('relu')(net_7)
net_8 = Conv2D(64, (1, 1), padding=padding)(net_6)
net_8 = Activation('relu')(net_8)
chunk_2 = concatenate([net_7, net_8], axis=3)

x = Flatten()(chunk_2)
x = Dropout(dropout_rate)(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(features, labels, epochs=32, validation_split=0.1, batch_size=128)