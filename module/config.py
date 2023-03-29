# directory
from __future__ import annotations
import datetime 
from module.config_tickers import DOW_30_TICKER

TODAY = datetime.datetime.today()
YESTERDAY = datetime.datetime.today() - datetime.timedelta(days=1)
DELTA = -(datetime.datetime.strptime('2009-01-01', '%Y-%M-%d') - YESTERDAY).days


DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = str(YESTERDAY.date() - datetime.timedelta(days=DELTA*0.2))

TEST_START_DATE = str(datetime.datetime.now)
TEST_END_DATE = str((YESTERDAY).date())

TRADE_START_DATE = str(YESTERDAY.date() - datetime.timedelta(days=DELTA*0.2))
TRADE_END_DATE = str(YESTERDAY.date())

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Environment
stock_dimension = len(DOW_30_TICKER)
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
buy_cost_list = sell_cost_list

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

# Model Parameters
A2C_PARAMS = {
    "n_steps": 2048, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0001
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.0001,
    "batch_size": 128,
}

DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.0001
    }

TD3_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 1000000, 
    "learning_rate": 0.0001
}

SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_SECRET = "4w4pWIn3L2LZ5bnSn0L5B4adzIielhmiYUVvMSO3"  # your ALPACA_API_SECRET
ALPACA_API_KEY = "PKG1FFR1MZBGN74CJ0EV" # your ALPACA_API_KEY
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
