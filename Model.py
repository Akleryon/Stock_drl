import sys
sys.path.append("../STOCK_DRL")

import warnings
warnings.filterwarnings("ignore")

from finrl.main import check_and_make_directories

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from Processed import get_processed_data

from finrl.plot import backtest_stats, get_baseline

from module.efficient_frontier import EfficientFrontier
from module import helper
from module.config_tickers import DOW_30_TICKER
from module.env_stocktrading import StockTradingEnv
from module.models import DRLAgent
from module.logger import configure
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

def modelling():
  print("========== CREATING DIRECTORIES ==========")

  check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

  if_using_a2c = True
  if_using_ddpg = True
  if_using_ppo = True
  if_using_td3 = True
  if_using_sac = True

  mvo_df, train, trade, stock_dimension, state_space, sell_cost_list, num_stock_shares, processed = get_processed_data(ticker= DOW_30_TICKER, indicators=INDICATORS)
  buy_cost_list = sell_cost_list

  fst = mvo_df
  fst = fst.iloc[0*29:0*29+29, :]
  tic = fst['tic'].tolist()

  mvo = pd.DataFrame()

  for k in range(len(tic)):
    mvo[tic[k]] = 0

  for i in range(mvo_df.shape[0]//29):
    n = mvo_df
    n = n.iloc[i*29:i*29+29, :]
    date = n['date'][i*29]
    mvo.loc[date] = n['close'].tolist()

  env_kwargs = {
      "hmax": 100,
      "initial_amount": 1000000,
      "num_stock_shares": num_stock_shares,
      "buy_cost_pct": buy_cost_list,
      "sell_cost_pct": sell_cost_list,
      "state_space": state_space,
      "stock_dim": stock_dimension,
      "tech_indicator_list": INDICATORS,
      "action_space": stock_dimension,
      "reward_scaling": 1e-4
  }

  PPO_PARAMS = {
      "n_steps": 2048,
      "ent_coef": 0.01,
      "learning_rate": 0.00025,
      "batch_size": 128,
  }

  TD3_PARAMS = {"batch_size": 100, 
                "buffer_size": 1000000, 
                "learning_rate": 0.001}

  SAC_PARAMS = {
      "batch_size": 128,
      "buffer_size": 100000,
      "learning_rate": 0.0001,
      "learning_starts": 100,
      "ent_coef": "auto_0.1",
  }


  e_train_gym = StockTradingEnv(df = train, **env_kwargs)
  env_train, _ = e_train_gym.get_sb_env()

  print(type(env_train))

  agent = DRLAgent(env = env_train)

  model_a2c = agent.get_model("a2c")
  model_ddpg = agent.get_model("ddpg")
  model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
  model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
  model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

  print("========== TRAINING MODELS ==========")

  if if_using_a2c:
    # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

  if if_using_ddpg:
    # set up logger
    tmp_path = RESULTS_DIR + '/ddpg'
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)

  if if_using_ppo:
    # set up logger
    tmp_path = RESULTS_DIR + '/ppo'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)

  if if_using_td3:
    # set up logger
    tmp_path = RESULTS_DIR + '/td3'
    newmodel_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

  if if_using_sac:
    # set up logger
    tmp_path = RESULTS_DIR + '/sac'
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)

  trained_a2c = agent.train_model(model=model_a2c, 
                              tb_log_name='a2c',
                              total_timesteps=50000) if if_using_a2c else None

  trained_ddpg = agent.train_model(model=model_ddpg, 
                              tb_log_name='ddpg',
                              total_timesteps=50000) if if_using_ddpg else None

  trained_ppo = agent.train_model(model=model_ppo, 
                              tb_log_name='ppo',
                              total_timesteps=50000) if if_using_ppo else None

  trained_td3 = agent.train_model(model=model_td3, 
                              tb_log_name='td3',
                              total_timesteps=50000) if if_using_td3 else None

  trained_sac = agent.train_model(model=model_sac, 
                              tb_log_name='sac',
                              total_timesteps=50000) if if_using_sac else None

  e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
  env_trade, obs_trade = e_trade_gym.get_sb_env()

  print("========== TESTING MODELS ==========")

  df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
      model=trained_a2c, 
      environment = e_trade_gym)

  df_account_value_ddpg, df_actions_a2c = DRLAgent.DRL_prediction(
      model=trained_ddpg, 
      environment = e_trade_gym)

  df_account_value_sac, df_actions_a2c = DRLAgent.DRL_prediction(
      model=trained_sac, 
      environment = e_trade_gym)

  df_account_value_ppo, df_actions_a2c = DRLAgent.DRL_prediction(
      model=trained_ppo, 
      environment = e_trade_gym)

  df_account_value_td3, df_actions_a2c = DRLAgent.DRL_prediction(
      model=trained_td3, 
      environment = e_trade_gym)

  print("========== COMPARAISON ==========")

  # Obtain optimal portfolio sets that maximize return and minimize risk

  #Dependencies

  #input k-portfolio 1 dataset comprising 15 stocks
  # StockFileName = './DJIA_Apr112014_Apr112019_kpf1.csv'

  Rows = 3551  #excluding header
  Columns = 15  #excluding date
  portfolioSize = 29 #set portfolio sizedatetime..tolist()
  # print(assetLabels)

  #extract asset prices
  # StockData = df.iloc[0:, 1:]
  StockData = mvo.head(mvo.shape[0]-658)
  TradeData = mvo.tail(658)
  # df.head()
  TradeData.to_numpy()

  #compute asset returns
  arStockPrices = np.asarray(StockData)
  [Rows, Cols]=arStockPrices.shape
  arReturns = helper.StockReturnsComputing(arStockPrices, Rows, Cols)


  #compute mean returns and variance covariance matrix of returns
  meanReturns = np.mean(arReturns, axis = 0)
  covReturns = np.cov(arReturns, rowvar=False)
  
  #set precision for printing results
  np.set_printoptions(precision=3, suppress = True)

  #display mean returns and variance-covariance matrix of returns
  print('Mean returns of assets in k-portfolio 1\n', meanReturns)
  print('Variance-Covariance matrix of returns\n', covReturns)

  ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
  raw_weights_mean = ef_mean.max_sharpe()
  cleaned_weights_mean = ef_mean.clean_weights()
  mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
  LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
  Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
  Portfolio_Assets = TradeData @ Initial_Portfolio
  MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

  df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
  df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
  df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
  df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
  df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])

  result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True, right_index=True)
  result = pd.merge(result, df_result_td3, left_index=True, right_index=True)
  result = pd.merge(result, df_result_ppo, left_index=True, right_index=True)
  result = pd.merge(result, df_result_sac, left_index=True, right_index=True)
  result = pd.merge(result, MVO_result, left_index=True, right_index=True)
  result.columns = ['a2c', 'ddpg', 'td3', 'ppo', 'sac', 'mean var']

  print("========== SAVING MODELS and GRAPH ==========")

  
  plt.rcParams["figure.figsize"] = (15,5)
  plt.figure()
  result.plot()
  plt.savefig("trained_models/models_" + str(datetime.datetime.now()) + ".jpg")
  
  baseline_df = get_baseline(
        ticker="^DJI", 
        start = TRADE_START_DATE,
        end = TRADE_END_DATE)

  print("==============Compare to DJIA===========")
  # S&P 500: ^GSPC
  # Dow Jones Index: ^DJI
  # NASDAQ 100: ^NDX
  helper.backtest_plot(df_account_value_sac, 
                baseline_ticker = '^DJI', 
                baseline_start = TRADE_START_DATE,
                baseline_end = TRADE_END_DATE)

  return trained_a2c, trained_ppo, trained_ddpg, trained_td3, trained_sac

