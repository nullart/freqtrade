from base64 import urlsafe_b64encode

import pytest
import simplejson as json
from freqtrade.aws.backtesting_lambda import backtest
from freqtrade.aws.strategy import submit


def test_backtest(lambda_context):
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyFancyTestStrategy(IStrategy):
    minimal_roi = {
        "0": 0.5
    }
    stoploss = -0.2
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['maShort'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maShort'], dataframe['maMedium'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maMedium'], dataframe['maShort'])
            ),
            'sell'] = 1
        return dataframe


        """

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "description": "simple test strategy",
        "name": "MyFancyTestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # now we add an entry
    submit({
        "body": json.dumps(request)
    }, {})

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "name": "MyFancyTestStrategy",
        "stake_currency": "usdt",
        "asset": ["ETH", "BTC", "XRP", "LTC"],
        "exchange": "binance"
    }

    backtest({"body": json.dumps(request)}, {})
