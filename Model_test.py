import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from stable_baselines3 import (
    PPO,
    TD3,
    SAC,
    A2C,
    DDPG
)
from Processed import get_processed_data
from module import helper
from module.yahoodownloader import YahooDownloader
from module.env_stocktrading import StockTradingEnv
from module.models import DRLAgent
from module.config import (
    INDICATORS,
    env_kwargs,
    A2C_PARAMS,
    PPO_PARAMS,
    SAC_PARAMS,
    DDPG_PARAMS,
    TD3_PARAMS,
    TRADE_START_DATE,
    TRADE_END_DATE
)
from module.config_tickers import (
    DOW_30_TICKER
)

mvo_df, train, trade, df = get_processed_data(
    ticker=DOW_30_TICKER, indicators=INDICATORS)

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

e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

print("========== TESTING MODELS ==========")

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=A2C.load(path='trained_models/a2c.zip', kwargs=A2C_PARAMS),
    environment=e_trade_gym)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=DDPG.load(path='trained_models/ddpg.zip', kwargs=DDPG_PARAMS),
    environment=e_trade_gym)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=SAC.load(path='trained_models/sac.zip', kwargs=SAC_PARAMS),
    environment=e_trade_gym)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=PPO.load(path='trained_models/ppo.zip', kwargs=PPO_PARAMS),
    environment=e_trade_gym)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=TD3.load(path='trained_models/td3.zip', kwargs=TD3_PARAMS),
    environment=e_trade_gym)

print("========== COMPARAISON ==========")

df_result_a2c = df_account_value_a2c.set_index(
    df_account_value_a2c.columns[0])
df_result_ddpg = df_account_value_ddpg.set_index(
    df_account_value_ddpg.columns[0])
df_result_td3 = df_account_value_td3.set_index(
    df_account_value_td3.columns[0])
df_result_ppo = df_account_value_ppo.set_index(
    df_account_value_ppo.columns[0])
df_result_sac = df_account_value_sac.set_index(
    df_account_value_sac.columns[0])

result=pd.merge(df_result_a2c, df_result_ddpg,left_index = True, right_index = True)
result=pd.merge(result, df_result_td3, left_index = True, right_index = True)
result=pd.merge(result, df_result_ppo, left_index = True, right_index = True)
result=pd.merge(result, df_result_sac, left_index = True, right_index = True)
result.columns=['a2c', 'ddpg', 'td3', 'ppo', 'sac']

print("========== SAVING MODELS and GRAPH ==========")

plt.rcParams["figure.figsize"]=(15, 5)
plt.figure()
result.plot()
plt.savefig("trained_models/models_" + ".jpg")

now=datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_sac=helper.backtest_stats(
    account_value = df_account_value_sac)
perf_stats_sac=pd.DataFrame(perf_stats_sac)

perf_stats_ddpg=helper.backtest_stats(
    account_value = df_account_value_ddpg)
perf_stats_ddpg=pd.DataFrame(perf_stats_ddpg)

perf_stats_ppo=helper.backtest_stats(
    account_value = df_account_value_ppo)
perf_stats_ppo=pd.DataFrame(perf_stats_ppo)

perf_stats_td3=helper.backtest_stats(
    account_value = df_account_value_td3)
perf_stats_td3=pd.DataFrame(perf_stats_td3)

perf_stats_a2c=helper.backtest_stats(
    account_value = df_account_value_a2c)
perf_stats_a2c=pd.DataFrame(perf_stats_a2c)

baseline_df=YahooDownloader(
    ticker_list = ["^DJI"],
    start_date = TRADE_START_DATE,
    end_date = TRADE_END_DATE).fetch_data()

stats=helper.backtest_stats(baseline_df, value_col_name = 'close')

Annual_return_sac=perf_stats_sac.T['Annual return'][0]
Annual_return_ppo=perf_stats_ppo.T['Annual return'][0]
Annual_return_td3=perf_stats_td3.T['Annual return'][0]
Annual_return_ddpg=perf_stats_ddpg.T['Annual return'][0]
Annual_return_a2c=perf_stats_a2c.T['Annual return'][0]
Annual_return=stats[0]

print("===== BEST MODEL =======")

d={'sac': Annual_return_sac, 'a2c': Annual_return_a2c,
     'td3': Annual_return_td3, 'ddpg': Annual_return_ddpg, 'ppo': Annual_return_ppo}
best_model = max(d.items(), key=lambda i: i[1])
