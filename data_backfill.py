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
    BACKFILL_DAYS = 3
    end_dt_backfill = date.today()
    start_dt_backfill = end_dt_backfill - timedelta(days=BACKFILL_DAYS)

    # start_dt_backfill_str = start_dt_backfill.strftime("%Y-%m-%d")
    # end_dt_backfill_str = end_dt_backfill.strftime("%Y-%m-%d")
  
    systemDetails = system_initialization()
    systemDetails.init_trading()
    callKite = kiteAPIs()

    tokenList = [256265] ## NIFTY INDEX
    tokenList.extend(callKite.get_instrument_active_tokens('CE',start_dt_backfill))
    tokenList.extend(callKite.get_instrument_active_tokens('PE',start_dt_backfill))

    # tokenList = [11803650,11816450,11830530,11844610,11855362,11866626,15700226,16014594,16032002, 28834050,11812866,11828482,11844354,11855106,14582786,15671810]

    df = callKite.getHistoricalData(start_dt_backfill,  end_dt_backfill, tokenList, BACKFILL_INTERVAL)


    # df = callKite.extract_data_from_db(start_dt_backfill, end_dt_backfill, BACKFILL_INTERVAL, 256265)
    # print(df.head())

    # df_new = callKite.convert_minute_data_interval(df,to_interval=3)
    # print(df_new.head())



    
