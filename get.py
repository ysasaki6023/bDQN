# -*- coding: utf-8 -*-
from zaifapi import *
import sys,json,datetime,os

key = ""
secret = ""
token = "BTC"
outFolder = "streamData"


while True:
    try:
        zaif = ZaifPublicStreamApi()
        prevTime = None
        outFile = None
        for item in zaif.execute("btc_jpy"):
            stamp = datetime.datetime.strptime(item["timestamp"],"%Y-%m-%d %H:%M:%S.%f")
            if not prevTime:
                prevTime = stamp.hour
            if not prevTime == stamp.hour:
                prevTime = stamp.hour
                if outFile:
                    outFile.close()
                    outFile = None
            if not outFile:
                if not os.path.exists(outFolder):
                    os.makedirs(outFolder)
                outFilePath = os.path.join(outFolder,stamp.strftime("%Y-%m-%d_%H-%M-%S.json"))
                outFile = open(outFilePath,"w")
                print "open new file: %s"%outFilePath
            print stamp
            json.dump(item,outFile)
            outFile.write("\n")
            outFile.flush()
    except:
        pass
