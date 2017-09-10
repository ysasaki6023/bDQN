# -*- coding: utf-8 -*-
import gym, gym.spaces,csv,h5py,argparse,os,glob
import numpy as np

from keras.models import Sequential,Input
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam,RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import initializers

from env2cls import Environment

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.20))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-3)
parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="models/save")
parser.add_argument("--inputData","-i",dest="inputData"  , type=str,default="convertedData/2017*.h5f")
parser.add_argument("--validateData","-v",dest="validateData"  , type=str,default="convertedData/2017-*_1[09]*.h5f")
parser.add_argument("--reload","-r",dest="reload"  , type=str,default=None)
parser.add_argument("--batchSize","-b",dest="batchSize"  , type=int,default=32)
parser.add_argument("--zdim","-z",dest="zdim"  , type=int,default=16)
parser.add_argument("--estLength",dest="estLength"  , type=int,default=1*60*5) # 5fps前提で、だいたい2分くらい先まで予測
parser.add_argument("--estStep",dest="estStep"  , type=int,default=10*5) # 20秒間の平均を計算
nLookbackLength = 1000
nLookbackSteps = 100
nLookbackRange = (-5000,5000)
nLookbackRangeStep = 50

args = parser.parse_args()

if not os.path.exists(args.saveFolder):
    os.makedirs(args.saveFolder)

inputPathList = glob.glob(args.inputData)
inputPathList = set(inputPathList)

validPathList = glob.glob(args.validateData)
validPathList = set(validPathList)

inputPathList = list( inputPathList - validPathList)
validPathList = list(validPathList)

print "training Data :",inputPathList
print " testing Data :",validPathList

# Get the environment and extract the number of actions.
env = Environment(inputPathList=inputPathList)

np.random.seed(123)
env.seed(123)

# Next, we build a very simple model.
sampleX,sampleT = env.getBatch(1,args.estLength,args.estStep,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep)
print sampleX.shape

regul = 1e-8
model = Sequential()
model.add(Conv2D(filters= 32,kernel_size=5,activation="relu",input_shape=sampleX.shape[1:]))
model.add(Conv2D(filters= 32,kernel_size=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,3)))

model.add(Conv2D(filters=128,kernel_size=5,activation="relu",input_shape=sampleX.shape[1:]))
model.add(Conv2D(filters=128,kernel_size=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,3)))
#model.add(BatchNormalization())

model.add(Conv2D(filters=256,kernel_size=5,activation="relu",input_shape=sampleX.shape[1:]))
model.add(Conv2D(filters=256,kernel_size=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,5)))

#model.add(Conv2D(filters=512,kernel_size=5,activation=LeakyReLU(alpha=0.1),input_shape=sampleX.shape[1:]))
#model.add(Conv2D(filters=512,kernel_size=5,activation=LeakyReLU(alpha=0.1)))
#model.add(MaxPooling2D(pool_size=(2,5)))

#model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(64,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))

model.add(Dense(2,kernel_regularizer=regularizers.l2(regul)))
model.add(Activation('softmax'))

#model.add(Dense(args.zdim,kernel_regularizer=regularizers.l2(regul)))
#model.add(LeakyReLU(alpha=0.1))

#model.add(Dense(sampleT.shape[1],kernel_regularizer=regularizers.l2(regul),bias_initializer=initializers.Constant(value=1.0)))

eachSteps = range(int(args.estLength/args.estStep))

#model.compile(optimizer = RMSprop(lr=args.learnRate),
model.compile(optimizer = Adam(lr=args.learnRate),
              loss      =  "binary_crossentropy",
              metrics   = ["accuracy"])
print(model.summary())

if args.reload:
    agent.load_weights(args.reload)

env_valid = Environment(inputPathList=validPathList)

cp_cb = ModelCheckpoint(filepath = args.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
tb_cb = TensorBoard(log_dir=args.saveFolder, histogram_freq=1)
lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-10, epsilon=1e-10, verbose=1)

model.fit_generator(generator=env.yieldBatch(args.batchSize,args.estLength,args.estStep,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep),
                    epochs=10000000000,
                    steps_per_epoch=500,
                    validation_data=env_valid.yieldBatch(args.batchSize,args.estLength,args.estStep,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep),
                    validation_steps=10,
                    max_queue_size = 30,
                    callbacks=[cp_cb,tb_cb,lr_cb]
                    )
