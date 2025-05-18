# /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/KiteConnect

# 0 16 * * * cd /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup && /opt/anaconda3/envs/KiteConnect/bin/python data_backfill.py >> /Users/shashankbaveja/Downloads/Personal/KiteConnectAPI/trading_setup/data_backfill_cron.log 2>&1

from IPython import embed;
from kiteconnect import KiteConnect, KiteTicker
import mysql
import mysql.connector as sqlConnector
import datetime
from selenium import webdriver
import os
from pyotp import TOTP
import ast
import time
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from myKiteLib import system_initialization
from myKiteLib import kiteAPIs
import logging
import json
from datetime import date, timedelta
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions

from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    BACKFILL_INTERVAL = 'minute'
    BACKFILL_DAYS = 60
    end_dt_backfill = date.today()
    start_dt_backfill = end_dt_backfill - timedelta(days=BACKFILL_DAYS)

    # start_dt_backfill_str = start_dt_backfill.strftime("%Y-%m-%d")
    # end_dt_backfill_str = end_dt_backfill.strftime("%Y-%m-%d")
  
    systemDetails = system_initialization()
    systemDetails.init_trading()
    callKite = kiteAPIs()

    tokenList = [256265] ## NIFTY INDEX
    tokenList.extend(callKite.get_instrument_active_tokens('CE',end_dt_backfill))
    tokenList.extend(callKite.get_instrument_active_tokens('PE',end_dt_backfill))

    # tokenList = [22113794, 22502146, 22585602, 22610178, 23080194, 23177730, 23197954, 23206914, 23290626, 23307522, 23453442, 23616514, 23633410, 23660546, 23673346, 23678466, 23834626, 23871490, 23881986, 23913474, 23919874, 23932418, 23947266, 23967746, 25078530, 25363714, 25392386, 25402626, 25720834, 25736450, 25753858, 25771778, 25775362, 26183170, 26199554, 26573058, 26959618, 27132674, 27191810, 27520770, 27655938, 27977218, 28917250, 28943874, 28953346, 28992514, 29250306, 29281794, 29372162, 29721602, 29810178, 30110210, 30504962, 30526466, 31714050, 32239106, 32835330, 32966658, 32972546, 34043650, 34080514, 34433282, 34681602, 35624706]

    df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)


    # df = callKite.extract_data_from_db(start_dt_backfill, end_dt_backfill, BACKFILL_INTERVAL, 256265)
    # print(df.head())

    # df_new = callKite.convert_minute_data_interval(df,to_interval=3)
    # print(df_new.head())



    
