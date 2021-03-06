import os
import asyncio
from datetime import datetime, timedelta

import aiohttp
import nest_asyncio
import sqlite3
import pandas as pd
import pandas_ta as ta
import numpy as np
import scipy.signal
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

from trader import Trader
from utils import get_popular_coins
nest_asyncio.apply()

file_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_path)

client = Trader().client
loop = asyncio.get_event_loop()

TIME_FRAMES = ['15m', '1h', '4h']


class SymbolData:
    """Start websocket for live klines and get historical klines that don't exist"""
    def __init__(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Getting symbol list')
        self.symbols = self.read_symbols()
        self.bad_symbols = []
        self.intervals = ['15m', '1h', '4h']
        self.latest_klines = {}
        self.data_dict = {}
        for s in self.symbols:
            self.data_dict[s] = {}
            self.latest_klines[s] = {}
            for interval in self.intervals:
                self.data_dict[s][interval] = []
                self.latest_klines[s][interval] = {}
        self.tf_dict = {
            '15m': 15,
            '1h': 60,
            '4h': 240,
        }
        self.bsm = BinanceSocketManager(client)
        self.conn_key = self.bsm.start_multiplex_socket(self.get_streams(), self.get_data)
        self.shutdown = False
        self.t = Thread(target=self.websocket_loop)
        self.t.setDaemon(True)
        self.t.start()

    @staticmethod
    async def read_symbols():
        conn = sqlite3.connect('symbols.db')
        curs = conn.cursor()
        symbol_tfs = {tab[0] for tab in curs.execute("select name from sqlite_master where type = 'table'").fetchall()}
        symbols = {symbol.split('_')[0] for symbol in symbol_tfs}
        return symbols if symbols else None


class SignalData:

    @classmethod
    async def return_dataframes(cls, symbol, event_loop):
        """Get complete dataframes for given symbol with signals data for all given timeframes
        Main method for initial data preparation"""
        return await cls.add_ta_data(await cls.get_original_data(symbol), event_loop)
        # return await cls.get_original_data(symbol)

    @classmethod
    async def get_original_data(cls, symbol):
        """Load price data from the database
        This has to happen first"""
        conn = sqlite3.connect('symbols.db')
        dfs = []
        for tf in TIME_FRAMES:
            query = f'SELECT * from {symbol}_{tf}'
            try:
                df = pd.read_sql_query(query, conn)
                dfs.append(df)
            except:
                continue
        conn.close()
        return dfs

    @classmethod
    async def add_ta_data(cls, dfs, event_loop):
        """Create columns for RSI, MACD(m/s/h), EMAs(20/50/200), Heiken Ashi
        These can all be run as coroutine tasks"""
        coroutines = [[],[],[]]
        for i, df in enumerate(dfs):
            coroutines[i].append(cls.get_rsi(df))
            coroutines[i].append(cls.get_macd(df))
            coroutines[i].append(cls.get_emas(df))
            coroutines[i].append(cls.get_heiken_ashi(df))
        # run coroutines with event loop and get return VALUES
        dfs = []
        for c in coroutines:
            dfs.append(event_loop.run_until_complete(asyncio.gather(*c)))
        comp_dfs = []
        for t in dfs:
            try:
                df = pd.concat([t[i] for i in range(len(t))], axis=1)
                comp_dfs.append(df)
            except:
                continue
        # return return values
        return comp_dfs

    @classmethod
    async def get_rsi(cls, df):
        return ta.rsi(df.close, 14)

    @classmethod
    async def get_macd(cls, df):
        try:
            return ta.macd(df.close, 12, 26, 9)
        except IndexError:
            pass

    @classmethod
    async def get_emas(cls, df):
        try:
            df['ema_20'], df['ema_50'] = ta.ema(df.close, 20), ta.ema(df.close, 50)
            if len(df) >= 288:
                df['ema_200'] = ta.ema(df.close, 200)
            else:
                df['ema_200'] = ta.ema(df.close, len(df.close) - 3)
            df = df.tail(88)
            return df[['ema_20', 'ema_50', 'ema_200']]
        except IndexError:
            pass

    @classmethod
    async def get_heiken_ashi(cls, df):
        try:
            df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            idx = df.index.name
            df.reset_index(inplace=True)

            for i in range(0, len(df)):
                if i == 0:
                    df.at[i, 'HA_Open'] = ((df._get_value(i, 'open') + df._get_value(i, 'close')) / 2)
                else:
                    df.at[i, 'HA_Open'] = ((df._get_value(i - 1, 'HA_Open') + df._get_value(i - 1, 'HA_Close')) / 2)
            if idx:
                df.set_index(idx, inplace=True)

            df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
            df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)

            return df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
        except KeyError:
            pass

    @classmethod
    async def check_df(cls, df, event_loop):
        coroutines = [cls.check_rsi(df), cls.check_macd(df), cls.check_heiken_ashi(df)]
        results = event_loop.run_until_complete(asyncio.gather(*coroutines))
        return results

    @classmethod
    async def check_rsi(cls, df):
        return 'RSI'

    @classmethod
    async def check_macd(cls, df):
        return False, True

    @classmethod
    async def check_heiken_ashi(cls, df):
        return 'HA'

    @classmethod
    async def main(cls, symbol, event_loop):
        dfs = await cls.return_dataframes(symbol, event_loop)
        coroutines = [cls.check_df(dfs[i], event_loop) for i in range(len(dfs))]
        results = event_loop.run_until_complete(asyncio.gather(*coroutines))
        return results

    @classmethod
    async def all_main(cls, event_loop):
        cos = []
        for s in asyncio.run(SymbolData.read_symbols()):
            cos.append(SignalData.main(s, loop))
        print(event_loop.run_until_complete(asyncio.gather(*cos)))


if __name__ == '__main__':
    start_time = datetime.now()
    asyncio.run(SignalData.all_main(loop))
    loop.close()
    print(f'took: ' + str(datetime.now() - start_time))
