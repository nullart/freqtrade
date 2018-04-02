# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from argparse import Namespace
from typing import Dict, Tuple, Any, List, Optional

import arrow
from pandas import DataFrame
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import exchange
from freqtrade.analyze import Analyze
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Bittrex
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade


logger = logging.getLogger(__name__)


class Backtesting(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.analyze = Analyze(self.config)
        exchange._API = Bittrex({'key': '', 'secret': ''})

    @staticmethod
    def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        timeframe = [
            (arrow.get(min(frame.date)), arrow.get(max(frame.date)))
            for frame in data.values()
        ]
        return min(timeframe, key=operator.itemgetter(0))[0], \
            max(timeframe, key=operator.itemgetter(1))[1]

    def _generate_text_table(self, data: Dict[str, List[Dict]], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = self.config.get('stake_currency')

        floatfmt = ('s', 'd', '.2f', '.8f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.currency == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_BTC.sum(),
                result.duration.mean(),
                len(result[result.profit_BTC > 0]),
                len(result[result.profit_BTC < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_BTC.sum(),
            results.duration.mean(),
            len(results[results.profit_BTC > 0]),
            len(results[results.profit_BTC < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt)

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ticker: List, trade_count_lock: Dict,
            stake_amount: float, max_open_trades: int) -> Optional[Trade]:
        trade = Trade(
            pair=pair,
            open_rate=buy_row.close,
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee=exchange.get_fee(),
        )

        # calculate win/lose forwards from buy point
        for sell_row in partial_ticker:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            if self.analyze.should_sell(
                    trade, sell_row.close, sell_row.date, sell_row.buy, sell_row.sell):
                trade.close(sell_row.close, date=sell_row.date)
                return trade
        return None

    def backtest(
            self, stake_amount: float, processed: Dict[str, Any], realistic: bool,
            max_open_trades: Optional[int] = 0, record: Optional[bool] = False) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param stake_amount: btc amount to use for each trade
        :param processed: a processed dictionary with format {pair, data}
        :param realistic: do we try to simulate realistic trades?
        :param max_open_trades: maximum number of concurrent trades (default: 0, disabled)
        :param record: records trades made (default: False)
        :return: DataFrame
        """
        headers = ['date', 'buy', 'open', 'close', 'sell']
        records = []
        trades = []
        trade_count_lock = {}
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0
            ticker_data = self.analyze.populate_sell_trend(
                self.analyze.populate_buy_trend(pair_data)
            )[headers]
            ticker = list(ticker_data.itertuples())

            lock_pair_until = None
            for index, row in enumerate(ticker):
                if realistic:
                    if lock_pair_until and row.date <= lock_pair_until:
                        continue
                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue

                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                trade_entry = self._get_sell_trade_entry(
                    pair, row, ticker[index + 1:], trade_count_lock, stake_amount, max_open_trades)

                if trade_entry:
                    lock_pair_until = trade_entry.close_date
                    trades.append((
                        trade_entry.pair,
                        trade_entry.close_profit,
                        trade_entry.calc_profit(),
                        (trade_entry.close_date - trade_entry.open_date).seconds // 60,
                    ))
                    if record:
                        # Note, need to be json.dump friendly
                        # record a tuple of pair, current_profit_percent,
                        # entry-date, close-date and duration
                        records.append((
                            trade_entry.pair,
                            trade_entry.close_profit,
                            trade_entry.open_date.strftime('%s'),
                            trade_entry.close_date.strftime('%s'),
                            index,
                            (trade_entry.close_date - trade_entry.open_date).seconds // 60,
                        ))
        # For now export inside backtest(), maybe change so that backtest()
        # returns a tuple like: (dataframe, records, logs, etc)
        if record and record.find('trades') >= 0:
            logger.info('Dumping backtest results')
            file_dump_json('backtest-result.json', records)
        labels = ['currency', 'profit_percent', 'profit_BTC', 'duration']
        return DataFrame.from_records(trades, columns=labels)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        if self.config.get('live'):
            logger.info('Downloading data for all pairs in whitelist ...')
            for pair in pairs:
                data[pair] = exchange.get_ticker_history(
                    pair, self.analyze.strategy.ticker_interval
                )
        else:
            logger.info('Using local backtesting data (using whitelist in given config) ...')
            data = optimize.load_data(
                self.analyze.strategy.ticker_interval,
                pairs=pairs,
                refresh_pairs=self.config.get('refresh_pairs', False),
                timerange=Arguments.parse_timerange(self.config.get('timerange')),
                datadir=self.config['datadir'],
            )

        # Ignore max_open_trades in backtesting, except realistic flag was passed
        if self.config.get('realistic_simulation', False):
            max_open_trades = self.config['max_open_trades']
        else:
            logger.info('Ignoring max_open_trades (realistic_simulation not set) ...')
            max_open_trades = 0

        preprocessed = self.analyze.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)
        logger.info(
            'Measuring data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        # Execute backtest and print results
        results = self.backtest(
            self.config.get('stake_amount'),
            preprocessed,
            self.config.get('realistic_simulation', False),
            max_open_trades,
            self.config.get('export'),
        )
        logger.info(
            '\n==================================== '
            'BACKTESTING REPORT'
            ' ====================================\n'
            '%s',
            self._generate_text_table(data, results)
        )


def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    return config


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Initialize configuration
    config = setup_configuration(args)
    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()
