# -*- coding: utf-8 -*-
import numpy as np
import gym, gym.spaces
import csv,h5py,argparse,os,glob,random
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2

class Environment(gym.Env):
#############
    def __init__(self,inputPathList, nSteps_per_episode=5*60*6, initialJPY=100000.,
                 priceRange=(300000.,1000000.),volumeRange=(0.,100.), moneyRange=(0.,1000000.)):

        self.action_space = gym.spaces.Discrete(3)
        self.nSteps_per_episode = nSteps_per_episode # fps=6くらい。5分くらいを見る
        self.inputPathList = inputPathList

        self.initialJPY = initialJPY

        # 規格化定数。どうもランダムにサンプルするときに使うらしい
        self.priceMin, self.priceMax  = priceRange
        self.volumeMin,self.volumeMax = volumeRange
        self.moneyMin, self.moneyMax  = moneyRange
        depth_price_min,  depth_price_max  = priceRange
        depth_volume_min, depth_volume_max = volumeRange
        tseries_min,  tseries_max  = priceRange
        n_depth   = 20
        n_tseries = 10+10+10
        print "price range =",self.priceMin,"-",self.priceMax

        self.observation_space = gym.spaces.Box(low =np.concatenate([np.array([self.priceMin,self.priceMin,self.priceMin,self.priceMin,self.priceMin,0.0,self.moneyMin]) , np.array([depth_price_min]*n_depth) , np.array([depth_volume_min]*n_depth) , np.array([tseries_min]*n_tseries)]),
                                                high=np.concatenate([np.array([self.priceMax,self.priceMax,self.priceMax,self.priceMax,self.priceMax,1.0,self.moneyMax]) , np.array([depth_price_max]*n_depth) , np.array([depth_volume_max]*n_depth) , np.array([tseries_max]*n_tseries)])) 
        # bitCoin取引価格範囲(現在価格), 左同, bitCoin取引量, 所有金額（円）

        pass


#############
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
        if action==2 and newVolume > EPSILON: # sell. 最後の最後は必ず売るようにする
            volume           = min(self.buyVolume, max_bids_volume) # 買いに出されている分もしくは、自分の持っているBitcoin分でキャップ
            preVolume        = self.buyVolume
            newVolume        = self.buyVolume - volume
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
            reward = (self.moneyIHave + self.buyVolume * max_bids_price - self.initialJPY) # 最後は一応、すべて売り切れると仮定。妥当性については確認必要
        else:
            reward = 0.

        self.buyPrice  = 0. # しばらく使わないことにしておく
        state  = np.concatenate([np.array([min_asks_price,max_bids_price,min_asks_volume,max_bids_volume,self.buyPrice,self.buyVolume,self.moneyIHave]), depth_asks_price , depth_bids_price , depth_asks_volume , depth_bids_volume, tseries])

        return state, reward, done, {}

#############
    def _reset(self):

        self.__loadFile( random.choice(self.inputPathList) ) # ランダムに1ファイルを読み込み。self系の変数にセット

        self.posIdx = np.random.randint(0,len(self.min_asks_price)-self.nSteps_per_episode)
        self.iSteps = 0

        self.buyVolume = 0.
        self.buyPrice  = 0.
        self.moneyIHave = self.initialJPY # 10万円でスタート

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

#############
    def __loadFile(self,inFilePath):
        # 1ファイル分データを読み込んでselfへセットする

        # データの読み込み
        self.data = h5py.File(inFilePath)
        self.min_asks_price  = self.data["min_asks_price"].value
        self.max_bids_price  = self.data["max_bids_price"].value
        self.min_asks_volume = self.data["min_asks_volume"].value
        self.max_bids_volume = self.data["max_bids_volume"].value
        self.depth_asks      = self.data["depth_asks"].value # time x lines x (price,volume)
        self.depth_bids      = self.data["depth_bids"].value # time x lines x (price,volume)
        self.tseries         = self.data["tseries"].value # time x (10+10+10)
        #print self.depth_asks[-50:,:,0]
        #print self.depth_asks[:,:,0].max(),self.depth_asks[:,:,0].min()
        #raw_input()

        ## 板情報の規格化。まず中心を揃えて、その後、最大が1.0になるようにする
        center = self.center = np.mean   ([np.min(self.depth_asks[:,:,0],axis=1),np.max(self.depth_bids[:,:,0],axis=1)],axis=0)
        width  = self.width  = np.maximum(np.max(self.depth_asks[:,:,0],axis=1)-center, center - np.min(self.depth_bids[:,:,0],axis=1))
        #self.depth_asks[:,:,0] = (self.depth_asks[:,:,0] - np.expand_dims(center,axis=1))/np.expand_dims(width,axis=1)
        #self.depth_bids[:,:,0] = (np.expand_dims(center,axis=1) - self.depth_bids[:,:,0])/np.expand_dims(width,axis=1)

        #numOfLines = self.depth_asks.shape[1]*2 # bidsとasks双方
        #depth_price_min  = np.array( [0.]  * numOfLines)
        #depth_price_max  = np.array( [1.]  * numOfLines)
        #depth_volume_min = np.array( [self.volumeMin] * numOfLines)
        #depth_volume_max = np.array( [self.volumeMax] * numOfLines)

        ## 時系列変化の規格化。板の規格化と同じ値を使う
        self.tseries = (self.tseries - np.expand_dims(center,axis=1) ) / np.expand_dims(width,axis=1)
        tseries_max = np.array( [np.max(self.tseries)] * self.tseries.shape[1])
        tseries_min = np.array( [np.min(self.tseries)] * self.tseries.shape[1])

        self.data.close()

        return

#############
    def yieldBatch(self,nBatch,nLength,nSteps,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep):
        while True:
            yield self.getBatch(nBatch,nLength,nSteps,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep)

#############
    def getBatch(self,nBatch,nLength,nSteps,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep):
        self.__loadFile( random.choice(self.inputPathList) ) # ランダムに1ファイルを読み込み。self系の変数にセット
        batchX = []
        batchT = []
        for i in range(nBatch):
            x,t = self.getOne(nLength,nSteps,nLookbackLength,nLookbackSteps,nLookbackRange,nLookbackRangeStep)
            batchX.append(np.expand_dims(x,axis=0))
            batchT.append(np.expand_dims(t,axis=0))
        return np.concatenate(batchX,axis=0),np.concatenate(batchT,axis=0)

#############
    def getOne(self,nLength,nSteps,nLookbackLength,nLookbackSteps,nLookbackRange=(-5000,5000),nLookbackRangeStep=5):

        nWidth = int(nLength/nSteps)

        # すでに読み込まれているファイルからピックアップ
        self.posIdx = np.random.randint(nLookbackLength,len(self.min_asks_price)-nSteps*nWidth)

        #dd = np.zeros((self.nLookbackLength,(nLookbackRange[1]-nLookbackRange[0])/nLookbackRangeStep))
        y1 = self.depth_asks[self.posIdx-nLookbackLength:self.posIdx,:,0].copy() # copyなしだと、上書きされる
        z1 = self.depth_asks[self.posIdx-nLookbackLength:self.posIdx,:,1].copy()
        y2 = self.depth_bids[self.posIdx-nLookbackLength:self.posIdx,:,0].copy()
        z2 = self.depth_bids[self.posIdx-nLookbackLength:self.posIdx,:,1].copy()

        yc = self.depth_bids[self.posIdx-1,0,0].copy()

        #print self.posIdx, nLookbackLength
        x1 = np.tile(np.linspace(0,nLookbackLength-1,nLookbackLength),(y1.shape[1],1)).transpose(1,0)
        x2 = np.tile(np.linspace(0,nLookbackLength-1,nLookbackLength),(y2.shape[1],1)).transpose(1,0)
        ybins = int((nLookbackRange[1]-nLookbackRange[0])/nLookbackRangeStep)
        xx = np.concatenate([x1.flatten()   ,x2.flatten()   ])
        #print "original"
        #print "getOne()"
        #print y1[-50:]
        #print y2[-50:]
        y1 -= yc
        y2 -= yc
        #print y1[-1,0], y2[-1,0]
        #print y1[-2,0], y2[-2,0]
        #print "y1,y2="
        #print y1[-50:]
        #print y2[-50:]
        #print y1[-50:].mean(),y2[-50:].mean()

        #print y1.mean(), y2.mean()
        #print (y1[-1,0]-yc).mean(), (y2[-1,0]-yc).mean()
        yy = np.concatenate([y1.flatten(),y2.flatten()])
        #yy = np.concatenate([y1.flatten()-yc,y2.flatten()-yc])
        zz = np.concatenate([z1.flatten()   ,-z2.flatten()  ])

        #print yy[zz>0].mean(),yy[zz<0].mean()

        #h, xedges, yedges = np.histogram2d(xx,yy,weights=zz,range=((0,nLookbackSteps-1),nLookbackRange),bins=(nLookbackSteps,ybins))
        #print ((0,nLookbackSteps-1),nLookbackRange)
        h, xedges, yedges = np.histogram2d(xx,yy,weights=zz,bins=(nLookbackSteps,ybins),range=((0,nLookbackLength-1),nLookbackRange))

        #fig = plt.figure(figsize=(16,10))
        #ax = fig.add_subplot(111)
        #X, Y = np.meshgrid(xedges, yedges)
        #print X
        #print Y
        #ax.pcolormesh(X, Y, h.T, vmin=-1, vmax=1,cmap="seismic")
        #fig.show()
        #raw_input()
        h = np.expand_dims(h,axis=2) # color dim

        split_price  = np.split(self.max_bids_price [self.posIdx:self.posIdx+nSteps*nWidth],nWidth)
        #split_volume = np.split(self.max_bids_volume[self.posIdx:self.posIdx+nSteps*nWidth],nSteps)

        split_price  = np.mean(np.concatenate([np.expand_dims(p,axis=0) for p in split_price] ,axis=0),axis=1)
        #split_volume = np.mean(np.concatenate(split_volume,axis=0),axis=1)

        split_price /= self.max_bids_price[self.posIdx]
        #split_volume = np.mean(np.concatenate(split_volume,axis=0),axis=1)

        t = split_price[-1] # 最後の一つだけを注目
        tt  = (t <1.).astype(np.int32) * 0
        tt += (t==1.).astype(np.int32) * 1
        tt += (t >1.).astype(np.int32) * 2
        tt = np_utils.to_categorical(tt,3)[0]

        return h,tt
