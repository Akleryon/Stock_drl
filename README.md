# Stock_drl

This project was born thanks to the AI4Finance Fundation repository (https://github.com/AI4Finance-Foundation)

## Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading

### Task :

We train different DRL agents for stock trading. This task is modeled as a Markov Decision Process (MDP), and the objective function is maximizing (expected) cumulative return.

We specify the state-action-reward as follows:

State s: The state space represents an agent's perception of the market environment. Just like a human trader analyzing various information, here our agent passively observes many features and learns by interacting with the market environment (usually by replaying historical data).

Action a: The action space includes allowed actions that an agent can take at each state. For example, a ∈ {−1, 0, 1}, where −1, 0, 1 represent selling, holding, and buying. When an action operates multiple shares, a ∈{−k, ..., −1, 0, 1, ..., k}, e.g.. "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively.

Reward function r(s, a, s′): Reward is an incentive for an agent to learn a better policy. For example, it can be the change of the portfolio value when taking a at state s and arriving at new state s', i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively

Market environment: 30 consituent stocks of Dow Jones Industrial Average (DJIA) index. Accessed at the starting date of the testing period.

The data for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.

The different agents are based on the library SB3 (https://github.com/DLR-RM/stable-baselines3)

The class Alpaca can be used to trade online using the API. Disclaimer: I am sharing codes for academic purpose. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.

### Environment :

Python3.8
