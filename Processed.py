import os
import sys
sys.path.append("../STOCK_DRL")

import warnings
warnings.filterwarnings('ignore')

from  module.yahoodownloader import YahooDownloader
from module.preprocessor import FeatureEngineer
from module import config
from module import config_tickers

from module.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_30_TICKER,
                     interval='1D').fetch_data()

                     