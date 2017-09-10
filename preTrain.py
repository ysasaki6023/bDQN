# -*- coding: utf-8 -*-
import numpy as np
import gym, gym.spaces
import csv,h5py,argparse,os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from rl.agents.sarsa import SarsaAgent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.20))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-3)
parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="models/save")
parser.add_argument("--inputFile","-i",dest="inputFile"  , type=str,default="test.h5py")
parser.add_argument("--reload","-r",dest="reload"  , type=str,default=None)
parser.add_argument("--batchSize","-b",dest="batchSize"  , type=int,default=3200)
parser.add_argument("--zdim","-z",dest="zdim"  , type=int,default=16)
parser.add_argument("--estLength",dest="estLength"  , type=int,default=1200) # 5fps前提で、だいたい2分くらい先まで予測
parser.add_argument("--estStep",,dest="estStep"  , type=int,default=50) # 10秒間の平均を計算

args = parser.parse_args()

if not os.path.exists(args.saveFolder):
    os.makedirs(args.saveFolder)

class Environment(gym.Env):
    def __init__(self,inFilePath):
        self.action_space = gym.spaces.Discrete(3)
        self.nSteps_per_episode = args.nStepsPerEpisode # fps=6くらいなので、これだと27分くらいの間隔でトレードしていることになる

        # データの読み込み
        self.data = h5py.File(inFilePath)
        self.min_asks_price  = self.data["min_asks_price"].value
        self.max_bids_price  = self.data["max_bids_price"].value
        self.min_asks_volume = self.data["min_asks_volume"].value
        self.max_bids_volume = self.data["max_bids_volume"].value
        self.depth_asks      = self.data["depth_asks"].value # time x lines x (price,volume)
        self.depth_bids      = self.data["depth_bids"].value # time x lines x (price,volume)
        self.tseries         = self.data["tseries"].value # time x (10+10+10)
        self._reset()

        # 規格化
        ## price, volumeは基本的に規格化はせずに、最大・最小値を探すに留める。これをいじると、リワードの計算がおかしくなる
        self.priceMin     = np.min(self.depth_bids[:,:,0])
        self.priceMax     = np.max(self.depth_asks[:,:,0])
        self.volumeMin    = 0.
        self.volumeMax    = max(np.max(self.depth_asks[:,:,1]),np.max(self.depth_bids[:,:,1]))
        self.moneyMin     = 0.
        self.moneyMax     = 1000000. # 百万円まで認める
        print "price range =",self.priceMin,"-",self.priceMax

        ## 板情報の規格化。まず中心を揃えて、その後、最大が1.0になるようにする
        center = np.mean   ([np.min(self.depth_asks[:,:,0],axis=1),np.max(self.depth_bids[:,:,0],axis=1)],axis=0)
        width  = np.maximum(np.max(self.depth_asks[:,:,0],axis=1)-center, center - np.min(self.depth_bids[:,:,0],axis=1))
        self.depth_asks[:,:,0] = (self.depth_asks[:,:,0] - np.expand_dims(center,axis=1))/np.expand_dims(width,axis=1)
        self.depth_bids[:,:,0] = (np.expand_dims(center,axis=1) - self.depth_bids[:,:,0])/np.expand_dims(width,axis=1)

        numOfLines = self.depth_asks.shape[1]*2 # bidsとasks双方
        depth_price_min  = np.array( [0.]  * numOfLines)
        depth_price_max  = np.array( [1.]  * numOfLines)
        depth_volume_min = np.array( [self.volumeMin] * numOfLines)
        depth_volume_max = np.array( [self.volumeMax] * numOfLines)

        ## 時系列変化の規格化。板の規格化と同じ値を使う
        self.tseries = (self.tseries - np.expand_dims(center,axis=1) ) / np.expand_dims(width,axis=1)
        tseries_max = np.array( [np.max(self.tseries)] * self.tseries.shape[1])
        tseries_min = np.array( [np.min(self.tseries)] * self.tseries.shape[1])

        self.observation_space = gym.spaces.Box(low =np.concatenate([np.array([self.priceMin,self.priceMin,self.priceMin,self.priceMin,self.priceMin,0.0,self.moneyMin]) , depth_price_min , depth_volume_min,tseries_min]),
                                                high=np.concatenate([np.array([self.priceMax,self.priceMax,self.priceMax,self.priceMax,self.priceMax,1.0,self.moneyMax]) , depth_price_max , depth_volume_max,tseries_max])) # bitCoin取引価格範囲(現在価格), 左同, bitCoin取引量, 所有金額（円）

        pass

    def _step(self,action): #buy, keep, sell
        assert self.action_space.contains(action)
        self.iSteps += 1

        idx = self.posIdx + self.iSteps
        min_asks_price   = self.min_asks_price [idx]
        min_asks_volume  = self.min_asks_volume[idx]
        max_bids_price   = self.max_bids_price [idx]
        max_bids_volume  = self.max_bids_volume[idx]
        depth_asks_price  = self.depth_asks[idx,:,0]
        depth_asks_volume = self.depth_asks[idx,:,1]
        depth_bids_price  = self.depth_asks[idx,:,0]
        depth_bids_volume = self.depth_bids[idx,:,1]
        tseries = self.tseries[idx]

        EPSILON = 10./500000. # 大体10円分を計算下限値とする

        done   = True if (self.iSteps+1)>self.nSteps_per_episode else False

        #reward = - args.penalty # 何もしないと損するようにペナルティを課す。絶対金額。
        if action==2: # sell. 最後の最後は必ず売るようにする
            volume           = min(self.buyVolume, max_bids_volume) # 買いに出されている分もしくは、自分の持っているBitcoin分でキャップ
            preVolume        = self.buyVolume
            newVolume        = self.buyVolume - volume
            if newVolume < EPSILON: newVolume = 0. # 念の為リセット。マイナスを売る、などということが起こらないようにする
            #if newVolume>0.:
            #    self.buyPrice  = (preVolume*self.buyPrice - volume*max_bids_price) / newVolume # 今保有しているBitCoinの平均購入価格
            #else:
            #    self.buyPrice  = 0.

            self.buyVolume   = newVolume
            self.moneyIHave += max_bids_price * volume
        elif action==1: # buy
            volume           = min(self.moneyIHave / min_asks_price, min_asks_volume) # 今、askされている量もしくは、自分の持っているお金で買える分、いずれか小さい方でキャップ
            preVolume        = self.buyVolume
            newVolume        = self.buyVolume + volume
            #self.buyPrice    = (preVolume*self.buyPrice + volume*min_asks_price) / newVolume # 今保有しているBitCoinの平均購入価格
            self.buyVolume   = newVolume
            self.moneyIHave -=  min_asks_price * volume
        elif action==0: # keep
            pass


        if done:
            reward = (self.moneyIHave + self.buyVolume * max_bids_price - args.initialJPY) # 最後は一応、すべて売り切れると仮定。妥当性については確認必要
        else:
            reward = 0.

        self.buyPrice  = 0. # しばらく使わないことにしておく
        state  = np.concatenate([np.array([min_asks_price,max_bids_price,min_asks_volume,max_bids_volume,self.buyPrice,self.buyVolume,self.moneyIHave]), depth_asks_price , depth_bids_price , depth_asks_volume , depth_bids_volume, tseries])

        return state, reward, done, {}

    def _reset(self):
        self.posIdx = np.random.randint(0,len(self.min_asks_price)-self.nSteps_per_episode)
        self.iSteps = 0

        self.buyVolume = 0.
        self.buyPrice  = 0.
        self.moneyIHave   = args.initialJPY # 10万円でスタート

        state  = np.concatenate([np.array([self.min_asks_price[self.posIdx],
                                 self.max_bids_price[self.posIdx],
                                 self.min_asks_volume[self.posIdx],
                                 self.max_bids_volume[self.posIdx],
                                 self.buyPrice,
                                 self.buyVolume,
                                 self.moneyIHave]),
                                 self.depth_asks[self.posIdx,:,0],
                                 self.depth_bids[self.posIdx,:,0],
                                 self.depth_asks[self.posIdx,:,1],
                                 self.depth_bids[self.posIdx,:,1],
                                 self.tseries[self.posIdx,:]],axis=0)

        return state

regul = 1e-8
# Get the environment and extract the number of actions.
env = Environment(inFilePath=args.inputFile)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Input(shape=(1,) + env.observation_space.shape)


model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

model.add(Dense(512,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(1024,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(512,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dense(256,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(128,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dense(32,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dense(args.zdim,kernel_regularizer=regularizers.l2(regul)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

if args.agent=="sarsa":
    policy = BoltzmannQPolicy()
    agent = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=100, policy=policy)
    agent.compile(Adam(lr=args.learnRate), metrics=['mae'])
elif args.agent=="dqn":
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.1/10./6.) # 10秒に1回くらいの頻度で新しいチャレンジをするチャンスを与える(assuming 6fps)。ただし、実際にチャレンジするのは10%くらいになってほしい。
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy, batch_size=args.batchSize, train_interval=args.trainInterval)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
elif args.agent=="dqnBoltz":
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy, batch_size=args.batchSize, train_interval=args.trainInterval)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
elif args.agent=="duel":
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.1/10./6.) # 10秒に1回くらいの頻度で新しいチャレンジをするチャンスを与える(assuming 6fps)。ただし、実際にチャレンジするのは10%くらいになってほしい。
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy, enable_dueling_network=True, dueling_type="avg", batch_size=args.batchSize, train_interval=args.trainInterval)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
elif args.agent=="duelBoltz":
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy, enable_dueling_network=True, dueling_type="avg", batch_size=args.batchSize, train_interval=args.trainInterval)
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
else:
    raise "unknown agent type"

if args.reload:
    agent.load_weights(args.reload)

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.rewards = {}
        self.actions = {}
        self.file = open(os.path.join(args.saveFolder,"result.csv"),"w")
        self.csv  = csv.writer(self.file)
        self.csv.writerow(["episode","reward","action0","action1","action2"])

    def on_episode_begin(self, episode, logs):
        self.rewards[episode] = []
        self.actions[episode] = [0,0,0]
        self.steprows = []

    def on_step_end(self, step, logs):
        episode   = logs["episode"]
        action    = logs["action"]
        #metricx   = logs["metrics"]
        reward    = logs["reward"]
        price_ask = logs["observation"][0]
        price_bid = logs["observation"][1]
        price_avg = 0.5*(price_ask+price_bid)
        price     = logs["observation"][4]
        volume    = logs["observation"][5]
        money     = logs["observation"][6]
        self.steprows.append([step,action,reward,price_ask,price_bid,price_avg,price,volume,money])
        self.actions[episode][action] += 1
        return

    def on_episode_end(self, episode, logs):
        episode_reward = logs['episode_reward']
        self.csv.writerow([episode,episode_reward,self.actions[episode][0],self.actions[episode][1],self.actions[episode][2]])
        self.file.flush()
        if args.saveFolder:
            agent.save_weights(os.path.join(args.saveFolder,'weights.%03d.h5f'%episode), overwrite=True)
        if args.saveFolder:
            with open(os.path.join(args.saveFolder,"log.%05d.csv"%episode),"w") as f:
                c = csv.writer(f)
                c.writerow(["step","action","reward","price_ask","price_bid","price_avg","price","volume","money"])
                c.writerows(self.steprows)

cb_ep = EpisodeLogger()
agent.fit(env, nb_steps=1000000000, visualize=False, verbose=2, callbacks=[cb_ep])

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=10, visualize=False, verbose=2)
