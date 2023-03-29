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
    PPO_PARAMS,
    SAC_PARAMS,
    A2C_PARAMS,
    DDPG_PARAMS,
    TD3_PARAMS,
    env_kwargs

)
from module.logger import configure
from module.models import DRLAgent
from module.env_stocktrading import StockTradingEnv
from module.config_tickers import DOW_30_TICKER
from Processed import get_processed_data
from finrl.main import check_and_make_directories
import warnings
import sys
sys.path.append("../STOCK_DRL")

warnings.filterwarnings("ignore")

print("========== CREATING DIRECTORIES ==========")

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

mvo_df, train, trade, df = get_processed_data(
    ticker=DOW_30_TICKER, indicators=INDICATORS)

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

print(type(env_train))

agent = DRLAgent(env=env_train)

model_a2c = agent.get_model("a2c", model_kwargs=A2C_PARAMS)
model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

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
        model_td3 = agent.get_model("sac", model_kwargs=SAC_PARAMS)

if if_using_sac:
        # set up logger
        tmp_path = RESULTS_DIR + '/sac'
        new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_sac.set_logger(new_logger_sac)

agent.train_model(model=model_a2c,
                      tb_log_name='a2c',
                      total_timesteps=30000) if if_using_a2c else None

agent.train_model(model=model_ddpg,
                      tb_log_name='ddpg',
                      total_timesteps=30000) if if_using_ddpg else None

agent.train_model(model=model_ppo,
                      tb_log_name='ppo',
                      total_timesteps=30000) if if_using_ppo else None

agent.train_model(model=model_td3,
                      tb_log_name='td3',
                      total_timesteps=30000) if if_using_td3 else None

agent.train_model(model=model_sac,
                      tb_log_name='sac',
                      total_timesteps=30000) if if_using_sac else None
