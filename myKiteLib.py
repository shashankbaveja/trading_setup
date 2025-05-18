import json
from kiteconnect import KiteConnect, KiteTicker
import mysql
import numpy as np
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
import requests
from IPython import embed
from kiteconnect.exceptions import KiteException  # Import KiteConnect exceptions
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

telegramToken = '8135376207:AAFoMWbyucyPPEzc7CYeAMTsNZfqHWYDMfw'
telegramChatId = "-4653665640"

class system_initialization:
    def __init__(self):
        
        self.Kite = None
        self.con = None

        config_file_path = "./security.txt"
        with open(config_file_path, 'r') as file:
            content = file.read()

        self.config = ast.literal_eval(content)
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
        self.userId = self.config["userID"]
        self.pwd = self.config["pwd"]
        self.totp_key = self.config["totp_key"]
        
        self.mysql_username = self.config["username"]
        self.mysql_password = self.config["password"]
        self.mysql_hostname = self.config["hostname"]
        self.mysql_port = int(self.config["port"])
        self.mysql_database_name = self.config["database_name"]
        self.AccessToken = self.config["AccessToken"]

        print("read security details")

        self.kite = KiteConnect(api_key=self.api_key)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
    
    
    def init_trading(self):
        try:
            data = self.kite.historical_data(256265,'2025-05-15','2025-05-15','minute')
            print(data)
        except KiteException as e:
            print(KiteException)
            print("Access token expired, Generating new token")
            temp_token = self.kite_chrome_login_generate_temp_token()
            AccessToken = self.kite.generate_session(temp_token, api_secret= self.api_secret)["access_token"]
            self.kite.set_access_token(AccessToken)
            self.SaveAccessToken(AccessToken)

            # Update the in-memory configuration and write to security.txt
            self.config["AccessToken"] = AccessToken # Update self.config dict
            config_file_path = "./security.txt"
            try:
                with open(config_file_path, 'w') as file:
                    json.dump(self.config, file, indent=4) # Write the whole updated config
                print("Successfully updated AccessToken in security.txt")
            except Exception as update_err:
                print(f"Error updating AccessToken in security.txt: {update_err}")
            
            AccessToken = self.GetAccessToken()
            self.kite.set_access_token(AccessToken)        

        df_nse = self.download_instruments('NSE')
        df_nfo = self.download_instruments('NFO')
        df=pd.concat([df_nse, df_nfo], axis=0)
        print("starting DB save")
        self.save_intruments_to_db(data = df)

        print('Ready to trade')
        telegramMessage = 'Ready to trade'
        telegramURL = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(telegramToken,telegramChatId,telegramMessage)
        response = requests.get(telegramURL)
        return self.kite

    def GetAccessToken(self):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        cursor = self.con.cursor()
        query = "Select token from daily_token_log where date(created_at) = '{}' order by created_at desc limit 1;".format(datetime.date.today())
        cursor.execute(query)
        print("read token from DB")
        for row in cursor:
            if row is None:
                return ''
            else:
                return row[0]
        self.con.close()
    def run_query_limit_1(self,query):
        cursor = self.con.cursor()
        cursor.execute(query)
        print("read token from DB")
        for row in cursor:
            if row is None:
                return ''
            else:
                return row[0]
    def SaveAccessToken(self,Token):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        q1 = "INSERT INTO kiteConnect.daily_token_log(token) Values('{}')"
        q2 = " ON DUPLICATE KEY UPDATE Token = '{}', created_at=CURRENT_TIMESTAMP();"
        q1 = q1.format(Token)
        q2 = q2.format(Token)
        query = q1 + q2
        cur = self.con.cursor()
        cur.execute(query)
        self.con.commit()
        print("saved token to DB")
        self.con.close()


    def kite_chrome_login_generate_temp_token(self):
        browser = webdriver.Chrome()
        browser.get(self.kite.login_url())
        browser.implicitly_wait(5)

        username = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
        password = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input') 
        
        username.send_keys(self.userId)
        password.send_keys(self.pwd)
        
        # Click Login Button
        browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()
        time.sleep(2)

        pin = browser.find_element("xpath", '/html/body/div[1]/div/div[2]/div[1]/div[2]/div/div[2]/form/div[1]/input')
        totp = TOTP(self.totp_key)
        token = totp.now()
        pin.send_keys(token)
        time.sleep(1)
        temp_token=browser.current_url.split('request_token=')[1][:32]
        browser.close()

        print("got temp token")
        
        return temp_token

    def download_instruments(self, exch):
        lst = []
        if exch == 'NSE':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_NSE)
        elif exch == 'NFO':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_NFO) # derivatives NSE
        elif exch == 'BSE':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BSE)
        elif exch =='CDS':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_CDS) # Currency
        elif exch == 'BFO':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BFO) # Derivatives BSE
        elif exch == 'MCX':
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_MCX) # Commodity
        else:
            lst = self.kite.instruments(exchange=self.kite.EXCHANGE_BCD)
            
        df = pd.DataFrame(lst) # Convert list to dataframe
        if len(df) == 0:
            print('No data returned')
            return
        print("downloading instruments")
        return df

    def save_data_to_db(self, data, tableName):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        print("starting DB save - entered function")
        engine = create_engine("mysql+pymysql://{user}:{pw}@{localhost}:{port}/{db}".format(user=self.mysql_username, localhost = self.mysql_hostname, port = self.mysql_port, pw=self.mysql_password, db=self.mysql_database_name))
        print("starting DB save - created engine")
        data.to_sql(tableName, con = engine, if_exists = 'replace', chunksize = 100000)
        print('Saved to Database')
        self.con.close()

    def save_intruments_to_db(self,data):
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        for i in range (0, len(data)):
            query = "insert into kiteConnect.instruments_zerodha values({},'{}','{}','{}',{},'{}',{},{},{},'{}','{}','{}') ON DUPLICATE KEY UPDATE instrument_token=instrument_token;".format(data['instrument_token'].iloc[i],data['exchange_token'].iloc[i],data['tradingsymbol'].iloc[i],data['name'].iloc[i],data['last_price'].iloc[i],data['expiry'].iloc[i],data['strike'].iloc[i],data['tick_size'].iloc[i],data['lot_size'].iloc[i],data['instrument_type'].iloc[i],data['segment'].iloc[i],data['exchange'].iloc[i])
            cur = self.con.cursor()
            cur.execute(query)
            self.con.commit()
            # print("saved token to DB")
        self.con.close()

class kiteAPIs:
    def __init__(self):
        self.Kite = None
        self.con = None
        self.startKiteSession = system_initialization()
        self.kite = self.startKiteSession.kite
        self.con = self.startKiteSession.con
        self.api_key = self.startKiteSession.api_key
        self.AccessToken = self.startKiteSession.GetAccessToken()
        self.kite.set_access_token(self.AccessToken)
        self.ticker = KiteTicker(self.startKiteSession.api_key, self.AccessToken)

        self.mysql_username = self.startKiteSession.mysql_username
        self.mysql_password = self.startKiteSession.mysql_password
        self.mysql_hostname = self.startKiteSession.mysql_hostname
        self.mysql_port = self.startKiteSession.mysql_port
        self.mysql_database_name = self.startKiteSession.mysql_database_name


    # getting the instrument token for a given symbol
    def get_instrument_token(self,symbol):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where tradingsymbol in ({symbol}) and instrument_type = 'EQ'"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        # if len(df) > 0:
        #     return int(df.iloc[0,0])
        # else:
        #     return -1
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    # getting all the instrument tokens for a given instrument type
    def get_instrument_all_tokens(self, instrument_type):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where instrument_type = '{instrument_type}'".format(instrument_type)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()
    
    def get_instrument_active_tokens(self, instrument_type, start_dt):
        query = f"SELECT instrument_token FROM kiteConnect.instruments_zerodha where expiry >= '{start_dt}' and instrument_type = '{instrument_type}'".format(instrument_type, start_dt)
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df['instrument_token'].values.astype(int).tolist()

    def extract_data_from_db(self, from_date, to_date, interval, instrument_token):
        query = f"SELECT a.*, b.strike FROM kiteConnect.historical_data_{interval} a left join kiteConnect.instruments_zerodha b on a.instrument_token = b.instrument_token where a.timestamp between '{from_date}' and '{to_date}' and a.instrument_token = {instrument_token}"
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')
        df = pd.read_sql(query, self.con)
        self.con.close()
        return df
    
    def convert_minute_data_interval(self, df, to_interval=1):
        if to_interval == 1:
            return df
        
        if df is None or df.empty:
            return pd.DataFrame() # Return empty DataFrame if input is empty

        if not isinstance(to_interval, int) or to_interval <= 0:
            raise ValueError("to_interval must be a positive integer.")

        # Ensure 'timestamp' column exists and is in datetime format
        # Assuming the datetime column is named 'timestamp' as per requirements.
        # If it's 'date' from getHistoricalData, it should be handled/renamed before this function
        # or this function should be adapted. For now, proceeding with 'timestamp'.
        if 'timestamp' not in df.columns:
            # Try to use 'date' column if 'timestamp' is missing, assuming it's the datetime column
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'}) 
            else:
                raise ValueError("DataFrame must contain a 'timestamp' or 'date' column for resampling.")
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f"Could not convert 'timestamp' column to datetime: {e}")

        # Sort by instrument_token and timestamp
        df = df.sort_values(by=['instrument_token', 'timestamp'])

        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'id': 'first', 
            'strike': 'first' 
        }

        all_resampled_dfs = []

        # Group by instrument_token and then by day for resampling
        # The pd.Grouper will use the 'timestamp' column, group by Day ('D'), using start_day as origin for daily grouping.
        grouped_by_token_day = df.groupby([
            'instrument_token', 
            pd.Grouper(key='timestamp', freq='D', origin='start_day')
        ])

        for (token, day_key), group_data in grouped_by_token_day:
            if group_data.empty:
                continue

            # Define the resampling origin for this specific day: 9:15 AM
            # day_key is the start of the day (00:00:00) from the Grouper
            origin_time_for_day = day_key + pd.Timedelta(hours=9, minutes=15)

            # Set timestamp as index for resampling this group
            group_data_indexed = group_data.set_index('timestamp')
            
            resampled_one_group = group_data_indexed.resample(
                rule=f'{to_interval}T',
                label='left', # Label of the interval is its start time
                origin=origin_time_for_day
            ).agg(agg_rules)

            # Drop rows where 'open' is NaN (meaning no data fell into this resampled interval)
            resampled_one_group = resampled_one_group.dropna(subset=['open'])

            if not resampled_one_group.empty:
                # Add instrument_token back as a column
                resampled_one_group['instrument_token'] = token
                all_resampled_dfs.append(resampled_one_group)
        
        if not all_resampled_dfs:
            return pd.DataFrame(columns=df.columns) # Return empty DF with original columns

        final_df = pd.concat(all_resampled_dfs)
        final_df = final_df.reset_index() # 'timestamp' becomes a column

        # Ensure final column order as per requirement
        # Desired order: ID, instrument_token, open, high, low, close, volume, strike, timestamp
        # Current columns likely: timestamp, open, high, low, close, volume, ID, strike, instrument_token
        
        # Define desired column order
        # (Make sure all these columns exist in final_df after aggregation and reset_index)
        # 'instrument_token' added above, 'timestamp' from reset_index
        desired_columns = ['ID', 'instrument_token', 'open', 'high', 'low', 'close', 'volume', 'strike', 'timestamp']
        
        # Filter out any columns that might not be present if original df was minimal
        # And reorder
        final_df_columns = [col for col in desired_columns if col in final_df.columns]
        final_df = final_df[final_df_columns]
        
        return final_df

    ## get data from kite API for a given token, from_date, to_date, interval
    def getHistoricalData(self, from_date, to_date, tokens, interval):
        # embed()
        if from_date > to_date:
            return
        
        if tokens == -1:
            print('Invalid symbol provided')
            return 'None'
        
        df = pd.DataFrame()

        i = 0

        token_exceptions = []
        for t in tokens:
            print(t)
            try:
                records = self.kite.historical_data(t, from_date=from_date, to_date=to_date, interval=interval)
                if len(records) > 0:
                    records = pd.DataFrame(records)
                    records['instrument_token'] = t
                    df = pd.concat([df, records], axis = 0)
            except KiteException as e:
                print(f"KiteConnect API error for token {t}: {str(e)}")
                token_exceptions.append(t)
                continue
            
        print(df.head())
        df['interval'] = interval
        df['id'] = df['instrument_token'].astype(str) + '_' + df['interval'] + '_' + df['date'].dt.strftime("%Y%m%d%H%M")
        
        self.con = sqlConnector.connect(host=self.mysql_hostname, user=self.mysql_username, password=self.mysql_password, database=self.mysql_database_name, port=self.mysql_port,auth_plugin='mysql_native_password')

        i = 0
        if interval == 'minute':
            for i in range (0, len(df)):
                query = "insert ignore into kiteConnect.historical_data_minute values('{}',{},'{}',{},{},{},{},{})".format(df['id'].iloc[i],df['instrument_token'].iloc[i],df['date'].iloc[i],df['open'].iloc[i],df['high'].iloc[i],df['low'].iloc[i],df['close'].iloc[i],df['volume'].iloc[i])
                print(i)
                cur = self.con.cursor()
                cur.execute(query)
                self.con.commit()
                # print("saved token to DB")
        elif interval == 'day':
            for i in range (0, len(df)):
                query = "insert ignore into kiteConnect.historical_data_day values('{}',{},'{}',{},{},{},{},{})".format(df['id'].iloc[i],df['instrument_token'].iloc[i],df['date'].iloc[i],df['open'].iloc[i],df['high'].iloc[i],df['low'].iloc[i],df['close'].iloc[i],df['volume'].iloc[i])
                print(i)
                cur = self.con.cursor()
                cur.execute(query)
                self.con.commit()
                # print("saved token to DB")
        elif interval == '5minute':
            for i in range (0, len(df)):
                query = "insert ignore into kiteConnect.historical_data_5minute values('{}',{},'{}',{},{},{},{},{})".format(df['id'].iloc[i],df['instrument_token'].iloc[i],df['date'].iloc[i],df['open'].iloc[i],df['high'].iloc[i],df['low'].iloc[i],df['close'].iloc[i],df['volume'].iloc[i])
                print(i)
                cur = self.con.cursor()
                cur.execute(query)
                self.con.commit()
                # print("saved token to DB")
        elif interval == '2minute':
            for i in range (0, len(df)):
                query = "insert ignore into kiteConnect.historical_data_2minute values('{}',{},'{}',{},{},{},{},{})".format(df['id'].iloc[i],df['instrument_token'].iloc[i],df['date'].iloc[i],df['open'].iloc[i],df['high'].iloc[i],df['low'].iloc[i],df['close'].iloc[i],df['volume'].iloc[i])
                print(i)
                cur = self.con.cursor()
                cur.execute(query)
                self.con.commit()
                # print("saved token to DB")
            
        self.con.close()
        print('token_exceptions: ',token_exceptions)
        return df
        

    
