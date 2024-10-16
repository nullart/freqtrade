# pragma pylint: disable=missing-docstring, C0103, bad-continuation, global-statement
# pragma pylint: disable=protected-access
import logging
from copy import deepcopy
from random import randint
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

import freqtrade.exchange as exchange
from freqtrade import OperationalException, DependencyException, TemporaryError
from freqtrade.exchange import (init, validate_pairs, buy, sell, get_balance, get_balances,
                                get_ticker, get_ticker_history, cancel_order, get_name, get_fee,
                                get_id, get_pair_detail_url, get_amount_lots)
from freqtrade.tests.conftest import log_has

API_INIT = False


def maybe_init_api(conf, mocker, force=False):
    global API_INIT
    if force or not API_INIT:
        mocker.patch('freqtrade.exchange.validate_pairs',
                     side_effect=lambda s: True)
        init(config=conf)
        API_INIT = True


def test_init(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    maybe_init_api(default_conf, mocker, True)
    assert log_has('Instance is running with dry_run enabled', caplog.record_tuples)


def test_init_exception(default_conf):
    default_conf['exchange']['name'] = 'wrong_exchange_name'

    with pytest.raises(
            OperationalException,
            match='Exchange {} is not supported'.format(default_conf['exchange']['name'])):
        init(config=default_conf)


def test_validate_pairs(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'LTC/BTC': '', 'XRP/BTC': '', 'NEO/BTC': ''
    })
    id_mock = PropertyMock(return_value='test_exchange')
    type(api_mock).id = id_mock

    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    validate_pairs(default_conf['exchange']['pair_whitelist'])


def test_validate_pairs_not_available(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    with pytest.raises(OperationalException, match=r'not available'):
        validate_pairs(default_conf['exchange']['pair_whitelist'])


def test_validate_pairs_not_compatible(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.load_markets = MagicMock(return_value={
        'ETH/BTC': '', 'TKN/BTC': '', 'TRST/BTC': '', 'SWT/BTC': '', 'BCC/BTC': ''
    })
    conf = deepcopy(default_conf)
    conf['stake_currency'] = 'ETH'
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', conf)
    with pytest.raises(OperationalException, match=r'not compatible'):
        validate_pairs(conf['exchange']['pair_whitelist'])


def test_validate_pairs_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    api_mock = MagicMock()
    api_mock.name = 'Binance'
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    api_mock.load_markets = MagicMock(return_value={})
    with pytest.raises(OperationalException, match=r'Pair ETH/BTC is not available at Binance'):
        validate_pairs(default_conf['exchange']['pair_whitelist'])

    api_mock.load_markets = MagicMock(side_effect=ccxt.BaseError())
    validate_pairs(default_conf['exchange']['pair_whitelist'])
    assert log_has('Unable to validate pairs (assuming they are correct). Reason: ',
                   caplog.record_tuples)


def test_validate_pairs_stake_exception(default_conf, mocker, caplog):
    caplog.set_level(logging.INFO)
    conf = deepcopy(default_conf)
    conf['stake_currency'] = 'ETH'
    api_mock = MagicMock()
    api_mock.name = 'binance'
    mocker.patch('freqtrade.exchange._API', api_mock)
    mocker.patch.dict('freqtrade.exchange._CONF', conf)

    with pytest.raises(
        OperationalException,
        match=r'Pair ETH/BTC not compatible with stake_currency: ETH'
    ):
        validate_pairs(default_conf['exchange']['pair_whitelist'])


def test_buy_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    order = buy(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'dry_run_buy_' in order['id']


def test_buy_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_buy_{}'.format(randint(0, 10 ** 6))
    api_mock.create_limit_buy_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    mocker.patch('freqtrade.exchange._API', api_mock)

    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    order = buy(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        mocker.patch('freqtrade.exchange._API', api_mock)
        buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(DependencyException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.InvalidOrder)
        mocker.patch('freqtrade.exchange._API', api_mock)
        buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(TemporaryError):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        buy(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(OperationalException):
        api_mock.create_limit_buy_order = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        buy(pair='ETH/BTC', rate=200, amount=1)


def test_sell_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    order = sell(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'dry_run_sell_' in order['id']


def test_sell_prod(default_conf, mocker):
    api_mock = MagicMock()
    order_id = 'test_prod_sell_{}'.format(randint(0, 10 ** 6))
    api_mock.create_limit_sell_order = MagicMock(return_value={
        'id': order_id,
        'info': {
            'foo': 'bar'
        }
    })
    mocker.patch('freqtrade.exchange._API', api_mock)

    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    order = sell(pair='ETH/BTC', rate=200, amount=1)
    assert 'id' in order
    assert 'info' in order
    assert order['id'] == order_id

    # test exception handling
    with pytest.raises(DependencyException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.InsufficientFunds)
        mocker.patch('freqtrade.exchange._API', api_mock)
        sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(DependencyException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.InvalidOrder)
        mocker.patch('freqtrade.exchange._API', api_mock)
        sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(TemporaryError):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        sell(pair='ETH/BTC', rate=200, amount=1)

    with pytest.raises(OperationalException):
        api_mock.create_limit_sell_order = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        sell(pair='ETH/BTC', rate=200, amount=1)


def test_get_balance_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    assert get_balance(currency='BTC') == 999.9


def test_get_balance_prod(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={'BTC': {'free': 123.4}})
    mocker.patch('freqtrade.exchange._API', api_mock)

    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    assert get_balance(currency='BTC') == 123.4

    with pytest.raises(OperationalException):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        get_balance(currency='BTC')


def test_get_balances_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    assert get_balances() == {}


def test_get_balances_prod(default_conf, mocker):
    balance_item = {
        'free': 10.0,
        'total': 10.0,
        'used': 0.0
    }

    api_mock = MagicMock()
    api_mock.fetch_balance = MagicMock(return_value={
        '1ST': balance_item,
        '2ST': balance_item,
        '3ST': balance_item
    })
    mocker.patch('freqtrade.exchange._API', api_mock)

    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    assert len(get_balances()) == 3
    assert get_balances()['1ST']['free'] == 10.0
    assert get_balances()['1ST']['total'] == 10.0
    assert get_balances()['1ST']['used'] == 0.0

    with pytest.raises(TemporaryError):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        get_balances()
    assert api_mock.fetch_balance.call_count == exchange.API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.fetch_balance = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        get_balances()
    assert api_mock.fetch_balance.call_count == 1


# This test is somewhat redundant with
# test_exchange_bittrex.py::test_exchange_bittrex_get_ticker
def test_get_ticker(default_conf, mocker):
    maybe_init_api(default_conf, mocker)
    api_mock = MagicMock()
    tick = {
        'symbol': 'ETH/BTC',
        'bid': 0.00001098,
        'ask': 0.00001099,
        'last': 0.0001,
    }
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    mocker.patch('freqtrade.exchange._API', api_mock)

    # retrieve original ticker
    ticker = get_ticker(pair='ETH/BTC')

    assert ticker['bid'] == 0.00001098
    assert ticker['ask'] == 0.00001099

    # change the ticker
    tick = {
        'symbol': 'ETH/BTC',
        'bid': 0.5,
        'ask': 1,
        'last': 42,
    }
    api_mock.fetch_ticker = MagicMock(return_value=tick)
    mocker.patch('freqtrade.exchange._API', api_mock)

    # if not caching the result we should get the same ticker
    # if not fetching a new result we should get the cached ticker
    ticker = get_ticker(pair='ETH/BTC')

    assert api_mock.fetch_ticker.call_count == 1
    assert ticker['bid'] == 0.5
    assert ticker['ask'] == 1

    assert 'ETH/BTC' in exchange._CACHED_TICKER
    assert exchange._CACHED_TICKER['ETH/BTC']['bid'] == 0.5
    assert exchange._CACHED_TICKER['ETH/BTC']['ask'] == 1

    # Test caching
    api_mock.fetch_ticker = MagicMock()
    get_ticker(pair='ETH/BTC', refresh=False)
    assert api_mock.fetch_ticker.call_count == 0

    with pytest.raises(TemporaryError):  # test retrier
        api_mock.fetch_ticker = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        get_ticker(pair='ETH/BTC', refresh=True)

    with pytest.raises(OperationalException):
        api_mock.fetch_ticker = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        get_ticker(pair='ETH/BTC', refresh=True)

    api_mock.fetch_ticker = MagicMock(return_value={})
    mocker.patch('freqtrade.exchange._API', api_mock)
    get_ticker(pair='ETH/BTC', refresh=True)


def make_fetch_ohlcv_mock(data):
    def fetch_ohlcv_mock(pair, timeframe, since):
        if since:
            assert since > data[-1][0]
            return []
        return data
    return fetch_ohlcv_mock


def test_get_ticker_history(default_conf, mocker):
    api_mock = MagicMock()
    tick = [
        [
            1511686200000,  # unix timestamp ms
            1,  # open
            2,  # high
            3,  # low
            4,  # close
            5,  # volume (in quote currency)
        ]
    ]
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))
    mocker.patch('freqtrade.exchange._API', api_mock)

    # retrieve original ticker
    ticks = get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1511686200000
    assert ticks[0][1] == 1
    assert ticks[0][2] == 2
    assert ticks[0][3] == 3
    assert ticks[0][4] == 4
    assert ticks[0][5] == 5

    # change ticker and ensure tick changes
    new_tick = [
        [
            1511686210000,  # unix timestamp ms
            6,  # open
            7,  # high
            8,  # low
            9,  # close
            10,  # volume (in quote currency)
        ]
    ]
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(new_tick))
    mocker.patch('freqtrade.exchange._API', api_mock)

    ticks = get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1511686210000
    assert ticks[0][1] == 6
    assert ticks[0][2] == 7
    assert ticks[0][3] == 8
    assert ticks[0][4] == 9
    assert ticks[0][5] == 10

    with pytest.raises(TemporaryError):  # test retrier
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        # new symbol to get around cache
        get_ticker_history('ABCD/BTC', default_conf['ticker_interval'])

    with pytest.raises(OperationalException):
        api_mock.fetch_ohlcv = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        # new symbol to get around cache
        get_ticker_history('EFGH/BTC', default_conf['ticker_interval'])


def test_get_ticker_history_sort(default_conf, mocker):
    api_mock = MagicMock()

    # GDAX use-case (real data from GDAX)
    # This ticker history is ordered DESC (newest first, oldest last)
    tick = [
        [1527833100000, 0.07666, 0.07671, 0.07666, 0.07668, 16.65244264],
        [1527832800000, 0.07662, 0.07666, 0.07662, 0.07666, 1.30051526],
        [1527832500000, 0.07656, 0.07661, 0.07656, 0.07661, 12.034778840000001],
        [1527832200000, 0.07658, 0.07658, 0.07655, 0.07656, 0.59780186],
        [1527831900000, 0.07658, 0.07658, 0.07658, 0.07658, 1.76278136],
        [1527831600000, 0.07658, 0.07658, 0.07658, 0.07658, 2.22646521],
        [1527831300000, 0.07655, 0.07657, 0.07655, 0.07657, 1.1753],
        [1527831000000, 0.07654, 0.07654, 0.07651, 0.07651, 0.8073060299999999],
        [1527830700000, 0.07652, 0.07652, 0.07651, 0.07652, 10.04822687],
        [1527830400000, 0.07649, 0.07651, 0.07649, 0.07651, 2.5734867]
    ]
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))
    mocker.patch('freqtrade.exchange._API', api_mock)

    # Test the ticker history sort
    ticks = get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1527830400000
    assert ticks[0][1] == 0.07649
    assert ticks[0][2] == 0.07651
    assert ticks[0][3] == 0.07649
    assert ticks[0][4] == 0.07651
    assert ticks[0][5] == 2.5734867

    assert ticks[9][0] == 1527833100000
    assert ticks[9][1] == 0.07666
    assert ticks[9][2] == 0.07671
    assert ticks[9][3] == 0.07666
    assert ticks[9][4] == 0.07668
    assert ticks[9][5] == 16.65244264

    # Bittrex use-case (real data from Bittrex)
    # This ticker history is ordered ASC (oldest first, newest last)
    tick = [
        [1527827700000, 0.07659999, 0.0766, 0.07627, 0.07657998, 1.85216924],
        [1527828000000, 0.07657995, 0.07657995, 0.0763, 0.0763, 26.04051037],
        [1527828300000, 0.0763, 0.07659998, 0.0763, 0.0764, 10.36434124],
        [1527828600000, 0.0764, 0.0766, 0.0764, 0.0766, 5.71044773],
        [1527828900000, 0.0764, 0.07666998, 0.0764, 0.07666998, 47.48888565],
        [1527829200000, 0.0765, 0.07672999, 0.0765, 0.07672999, 3.37640326],
        [1527829500000, 0.0766, 0.07675, 0.0765, 0.07675, 8.36203831],
        [1527829800000, 0.07675, 0.07677999, 0.07620002, 0.076695, 119.22963884],
        [1527830100000, 0.076695, 0.07671, 0.07624171, 0.07671, 1.80689244],
        [1527830400000, 0.07671, 0.07674399, 0.07629216, 0.07655213, 2.31452783]
    ]
    type(api_mock).has = PropertyMock(return_value={'fetchOHLCV': True})
    api_mock.fetch_ohlcv = MagicMock(side_effect=make_fetch_ohlcv_mock(tick))
    mocker.patch('freqtrade.exchange._API', api_mock)

    # Test the ticker history sort
    ticks = get_ticker_history('ETH/BTC', default_conf['ticker_interval'])
    assert ticks[0][0] == 1527827700000
    assert ticks[0][1] == 0.07659999
    assert ticks[0][2] == 0.0766
    assert ticks[0][3] == 0.07627
    assert ticks[0][4] == 0.07657998
    assert ticks[0][5] == 1.85216924

    assert ticks[9][0] == 1527830400000
    assert ticks[9][1] == 0.07671
    assert ticks[9][2] == 0.07674399
    assert ticks[9][3] == 0.07629216
    assert ticks[9][4] == 0.07655213
    assert ticks[9][5] == 2.31452783


def test_cancel_order_dry_run(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)

    assert cancel_order(order_id='123', pair='TKN/BTC') is None


# Ensure that if not dry_run, we should call API
def test_cancel_order(default_conf, mocker):
    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    api_mock = MagicMock()
    api_mock.cancel_order = MagicMock(return_value=123)
    mocker.patch('freqtrade.exchange._API', api_mock)
    assert cancel_order(order_id='_', pair='TKN/BTC') == 123

    with pytest.raises(TemporaryError):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == exchange.API_RETRY_COUNT + 1

    with pytest.raises(DependencyException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.InvalidOrder)
        mocker.patch('freqtrade.exchange._API', api_mock)
        cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == exchange.API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.cancel_order = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        cancel_order(order_id='_', pair='TKN/BTC')
    assert api_mock.cancel_order.call_count == 1


def test_get_order(default_conf, mocker):
    default_conf['dry_run'] = True
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    order = MagicMock()
    order.myid = 123
    exchange._DRY_RUN_OPEN_ORDERS['X'] = order
    assert exchange.get_order('X', 'TKN/BTC').myid == 123

    default_conf['dry_run'] = False
    mocker.patch.dict('freqtrade.exchange._CONF', default_conf)
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock(return_value=456)
    mocker.patch('freqtrade.exchange._API', api_mock)
    assert exchange.get_order('X', 'TKN/BTC') == 456

    with pytest.raises(TemporaryError):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.NetworkError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == exchange.API_RETRY_COUNT + 1

    with pytest.raises(DependencyException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.InvalidOrder)
        mocker.patch('freqtrade.exchange._API', api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == exchange.API_RETRY_COUNT + 1

    with pytest.raises(OperationalException):
        api_mock.fetch_order = MagicMock(side_effect=ccxt.BaseError)
        mocker.patch('freqtrade.exchange._API', api_mock)
        exchange.get_order(order_id='_', pair='TKN/BTC')
    assert api_mock.fetch_order.call_count == 1


def test_get_name(default_conf, mocker):
    mocker.patch('freqtrade.exchange.validate_pairs',
                 side_effect=lambda s: True)
    default_conf['exchange']['name'] = 'binance'
    init(default_conf)

    assert get_name() == 'Binance'


def test_get_id(default_conf, mocker):
    mocker.patch('freqtrade.exchange.validate_pairs',
                 side_effect=lambda s: True)
    default_conf['exchange']['name'] = 'binance'
    init(default_conf)

    assert get_id() == 'binance'


def test_get_pair_detail_url(default_conf, mocker):
    mocker.patch('freqtrade.exchange.validate_pairs',
                 side_effect=lambda s: True)
    default_conf['exchange']['name'] = 'binance'
    init(default_conf)

    url = get_pair_detail_url('TKN/ETH')
    assert 'TKN' in url
    assert 'ETH' in url

    url = get_pair_detail_url('LOOONG/BTC')
    assert 'LOOONG' in url
    assert 'BTC' in url

    default_conf['exchange']['name'] = 'bittrex'
    init(default_conf)

    url = get_pair_detail_url('TKN/ETH')
    assert 'TKN' in url
    assert 'ETH' in url

    url = get_pair_detail_url('LOOONG/BTC')
    assert 'LOOONG' in url
    assert 'BTC' in url


def test_get_fee(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.calculate_fee = MagicMock(return_value={
        'type': 'taker',
        'currency': 'BTC',
        'rate': 0.025,
        'cost': 0.05
    })
    mocker.patch('freqtrade.exchange._API', api_mock)
    assert get_fee() == 0.025


def test_get_amount_lots(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.amount_to_lots = MagicMock(return_value=1.0)
    mocker.patch('freqtrade.exchange._API', api_mock)
    assert get_amount_lots('LTC/BTC', 1.54) == 1
