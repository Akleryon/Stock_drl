from Alpaca import Alpaca
from stable_baselines3 import (
    PPO,
    SAC,
    TD3,
    DDPG,
    A2C
)
from module.config import (
    PPO_PARAMS,
    A2C_PARAMS,
    DDPG_PARAMS,
    TD3_PARAMS,
    SAC_PARAMS
)

model = PPO.load('trained_models/PPO/PPO_660k.zip', kwargs=PPO_PARAMS)
test = Alpaca(model=model)
test.run()

