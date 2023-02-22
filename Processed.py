import os
import sys
sys.path.append("../STOCK_DRL")

import warnings
warnings.filterwarnings('ignore')

import itertools
import pandas as pd

from  module.yahoodownloader import YahooDownloader
from module.preprocessor import FeatureEngineer, data_split

from module.config import (
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

def get_processed_data(interval, ticker, indicators):

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = indicators,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)


    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TRADE_END_DATE,
                        ticker_list = ticker,
                        interval=interval).fetch_data()

    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)
    mvo_df = processed_full.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]

    train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
    return train, trade, stock_dimension, state_space, buy_cost_list, num_stock_shares