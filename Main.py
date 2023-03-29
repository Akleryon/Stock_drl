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

model = SAC.load(path = 'trained_models/sac.zip', kwargs= SAC_PARAMS)

test = Alpaca(model=model)
test.run()

