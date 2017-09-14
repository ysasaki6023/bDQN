# -*- coding: utf-8 -*-
import gym, gym.spaces,csv,h5py,argparse,os,glob
import numpy as np

from keras.models import Sequential,Input
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import initializers

from env import Environment

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.20))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-3)
parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="models/save")
parser.add_argument("--inputData","-i",dest="inputData"  , type=str,default="convertedData/2017*.h5f")
parser.add_argument("--validateData","-v",dest="validateData"  , type=str,default="convertedData/2017-*_1[09]*.h5f")
parser.add_argument("--reload","-r",dest="reload"  , type=str,default=None)
parser.add_argument("--batchSize","-b",dest="batchSize"  , type=int,default=512)
parser.add_argument("--zdim","-z",dest="zdim"  , type=int,default=16)
parser.add_argument("--estLength",dest="estLength"  , type=int,default=5*60*5) # 5fps前提で、だいたい4分くらい先まで予測
parser.add_argument("--estStep",dest="estStep"  , type=int,default=20*5) # 20秒間の平均を計算
parser.add_argument("--showStep",dest="showStep"  , type=int,default=3) # 3つおきくらいに表示

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
sampleX,sampleT = env.getBatch(args.estLength,args.estStep,args.batchSize)

regul = 1e-8
model = Sequential()
model.add(Reshape(target_shape=(-1,),input_shape=sampleX.shape[1:]))

#model.add(Dense(1024,kernel_regularizer=regularizers.l2(regul)))
#model.add(LeakyReLU(alpha=0.1))

#model.add(Dense(512,kernel_regularizer=regularizers.l2(regul)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(BatchNormalization())

#model.add(Dense(256,kernel_regularizer=regularizers.l2(regul)))
#model.add(LeakyReLU(alpha=0.1))
#model.add(BatchNormalization())

#model.add(Dropout(0.5))
model.add(Dense(128,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dense(32,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(args.zdim,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(sampleT.shape[1],kernel_regularizer=regularizers.l2(regul),bias_initializer=initializers.Constant(value=1.0)))
model.add(Activation('linear'))

def eachError(n):
    def rel_error_at(t,y):
        v1 = K.sqrt(K.mean(K.square(t[:,n]-y[:,n])))
        v0 = K.sqrt(K.mean(K.square(t[:,n]-1.0   )))
        return v1/v0
    return rel_error_at
eachSteps = range(int(args.estLength/args.estStep))

model.compile(optimizer = Adam(lr=args.learnRate),
              loss      =  "mean_squared_error",
              metrics   = ["mean_absolute_error"]+[eachError(n) for n in eachSteps[::args.showStep]])
print(model.summary())

if args.reload:
    model.load_weights(args.reload)

env_valid = Environment(inputPathList=validPathList)

cp_cb = ModelCheckpoint(filepath = args.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
tb_cb = TensorBoard(log_dir=args.saveFolder, histogram_freq=1)
lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-10, epsilon=1e-10, verbose=1)

"""
model.fit_generator(generator=env.yieldBatch(args.estLength,args.estStep,args.batchSize),
                    epochs=10000000000,
                    steps_per_epoch=500,
                    validation_data=env_valid.yieldBatch(args.estLength,args.estStep,args.batchSize),
                    validation_steps=10,
                    max_queue_size = 30,
                    callbacks=[cp_cb,tb_cb,lr_cb]
                    )
"""
totFig = 0.
totCnt = 0
while True:
    x,t = env_valid.getBatch(args.estLength,args.estStep,1)
    y   = model.predict(x,batch_size=1)

    gon  = ((t>1.).astype(bool) & (y>1.).astype(bool))
    gon += ((t<1.).astype(bool) & (y<1.).astype(bool))
    avg = np.mean(gon)
    totFig += avg
    totCnt += 1
    totAvg  = totFig/totCnt
    print totAvg,avg

