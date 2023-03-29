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

def get_processed_data(ticker, indicators):

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = indicators,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)

    print("========== DOWNLOADING ==========")

    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TRADE_END_DATE,
                        ticker_list = ticker).fetch_data()

    print("========== PROCESSING ==========")

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

    print("========== PROCESSING OVER ==========")
    
    return mvo_df, train, trade, processed_full