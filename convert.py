# -*- coding: utf-8 -*-
import json,os,sys,h5py,glob,datetime
from collections import deque
import numpy as np

inFileGlob    = "streamData/2017-*.json"
outFileFold   = "convertedData"

deque_length = 1000 * 100
deque_price  = deque(maxlen=deque_length) # 1秒間に100回更新されて、それが1000秒続いても大丈夫なように設計
deque_time   = deque(maxlen=1000) # 直近100回の平均でfpsを計算
deque_fps    = deque(maxlen=1000) # 直近100回の平均でfpsを計算

nTimeSeries = 10

minimumLength = 6*1000 # 最低でも1000秒くらいはデータあるものを対象とする

avgFPS = 6.0
alpha  = 1e-2

if not os.path.exists(outFileFold):
    os.makedirs(outFileFold)

for inFilePath in sorted(glob.glob(inFileGlob)):
    outputFileName = os.path.join(outFileFold,os.path.basename(inFilePath).split(".")[0]+".h5f")
    print "converting :",inFilePath," -> ",outputFileName
    array_min_asks_price, array_max_bids_price, array_min_asks_volume, array_max_bids_volume = [],[],[],[]
    depth_asks,depth_bids = [],[]
    tseries_1s,tseries_10s,tseries_100s = [],[],[]

    with open(inFilePath) as inFile:
        for line in inFile:
            line = json.loads(line)
            # set deque
            time = datetime.datetime.strptime(line["timestamp"],"%Y-%m-%d %H:%M:%S.%f")
            deque_time.append(time)

            last_price = np.mean([ x["price"] for x in line["trades"]]) # 成約の平均をとっておく
            #print last_price

            if len(deque_price) < deque_length:
                deque_price.extend([last_price]*deque_length)
            else:
                deque_price.append(last_price)
            min_asks = sorted(line["asks"],key=lambda x:x[0])[::+1]
            max_bids = sorted(line["bids"],key=lambda x:x[0])[::-1]
            min_asks_price, min_asks_volume = min_asks[0]
            max_bids_price, max_bids_volume = max_bids[0]
            array_min_asks_price.append(min_asks_price)
            array_max_bids_price.append(max_bids_price)
            array_min_asks_volume.append(min_asks_volume)
            array_max_bids_volume.append(max_bids_volume)

            depth_asks.append(min_asks)
            depth_bids.append(max_bids)

            if len(deque_time)>1:
                timediff = deque_time[-1] - deque_time[0]
                secdiff  = timediff.seconds + timediff.microseconds/1000000.
                fps = float(len(deque_time))/secdiff
            else:
                fps = 10.

            avgFPS = (1.-alpha)*avgFPS + alpha*fps
            if avgFPS<1.: avgFPS = 1.

            list_deque_price = list(deque_price)
            past_prices_1s   = list_deque_price[::-int(  1*avgFPS)][:nTimeSeries]
            past_prices_10s  = list_deque_price[::-int( 10*avgFPS)][:nTimeSeries]
            past_prices_100s = list_deque_price[::-int(100*avgFPS)][:nTimeSeries]


            tseries_1s.append  ( np.array(past_prices_1s)  )
            tseries_10s.append ( np.array(past_prices_10s) )
            tseries_100s.append( np.array(past_prices_100s))
            tseries_1s[-1][min(tseries_1s[-1].shape[0],nTimeSeries):] = past_prices_1s[-1]

    if len(array_min_asks_price)<minimumLength:
        print ".... not enough length."
        continue

    array_min_asks_price = np.array(array_min_asks_price)
    array_max_bids_price = np.array(array_max_bids_price)
    array_min_asks_volume = np.array(array_min_asks_volume)
    array_max_bids_volume = np.array(array_max_bids_volume)

    depth_asks = np.array(depth_asks)
    depth_bids = np.array(depth_bids)

    tseries_1s   = np.stack(tseries_1s,axis=0)
    tseries_10s  = np.stack(tseries_10s,axis=0)
    tseries_100s = np.stack(tseries_100s,axis=0)

    outFile = h5py.File(outputFileName, "w")
    outFile.create_dataset("min_asks_price" ,data=array_min_asks_price)
    outFile.create_dataset("min_asks_volume",data=array_min_asks_volume)
    outFile.create_dataset("max_bids_price" ,data=array_max_bids_price)
    outFile.create_dataset("max_bids_volume",data=array_max_bids_volume)
    outFile.create_dataset("depth_asks"     ,data=depth_asks)
    outFile.create_dataset("depth_bids"     ,data=depth_bids)
    outFile.create_dataset("tseries"        ,data=np.concatenate([tseries_1s,tseries_10s,tseries_100s],axis=1))

    outFile.close()
