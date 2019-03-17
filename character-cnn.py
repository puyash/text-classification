
from datetime import datetime
from keras.layers import Dense, Input, Embedding, Dropout, Conv1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from functions import *


# conf and preprocess -----------------------------------------
# -------------------------------------------------------------

# settings ---------------------
# ------------------------------

EMBEDDING = False
TYPE = 'embedding' if EMBEDDING else 'standard'
MODELPATH ='models/char-conv-' + TYPE + '-{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5'
FILTERS = 500
LR = 0.0001 if EMBEDDING else 0.00001

CONV = [
    {'filters':500, 'kernel':8, 'strides':1, 'padding':'same', 'reg': 0, 'pool':2},
    {'filters':500, 'kernel':8, 'strides':1, 'padding':'same', 'reg': 0, 'pool':2},
    {'filters':500, 'kernel':8, 'strides':1, 'padding':'same', 'reg': 0, 'pool':''}
]



# generate dataset -------------
# ------------------------------

data, table = load_processed_data(False, not EMBEDDING)
print("input shape: ", np.shape(data.x_train))



# model architecture ------------------------------------------
# -------------------------------------------------------------


# input and embedding ----------
# ------------------------------

if EMBEDDING:

    inputlayer = Input(shape=(250,))
    network = Embedding(70, 16, input_length=250)(inputlayer)

else:
    inputlayer = Input(shape=(250 ,70))
    network = inputlayer

# convolutional layers ---------
# ------------------------------

for C in CONV:

    # conv layer
    network = Conv1D(filters=C['filters'], kernel_size=C['kernel'], \
                     strides=C['strides'], padding=C['padding'], activation='relu', \
                     kernel_regularizer=regularizers.l2(C['reg']))(network)

    if type(C['pool']) != int:
        continue

    # pooling layer
    network = MaxPooling1D(C['pool'])(network)

# fully connected --------------
# ------------------------------
network = Flatten()(network)
network = Dense(1024, activation='relu')(network)
network = Dropout(0)(network)

# output
ypred = Dense(2, activation='softmax')(network)


# training ----------------------------------------------------
# -------------------------------------------------------------


# callbacks --------------------
# ------------------------------

# tensorboard

TB_DIR = 'logs/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '_' + TYPE

os.makedirs(TB_DIR)
tensorboard = TensorBoard(log_dir=TB_DIR)

# early stopping and checkpoint
estopping = EarlyStopping(monitor='val_acc', patience=10)
checkpoint = ModelCheckpoint(filepath=MODELPATH, save_best_only=True)

# model-------------------------
# ------------------------------

optimizer = RMSprop(lr=LR)


model = Model(inputs=inputlayer, outputs=ypred)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])

print(TB_DIR)
print(model.summary())

# fit and run ------------------
# ------------------------------
try:
    hist = model.fit(data.x_train,
                     data.y_train,
                     validation_data=(data.x_test, data.y_test),
                     epochs=500,
                     batch_size=50,
                     shuffle=False,
                     verbose=2,
                     callbacks=[checkpoint, estopping, tensorboard])

except KeyboardInterrupt:    
    pass