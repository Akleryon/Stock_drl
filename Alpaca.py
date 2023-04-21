#from module.models import DRLAgent
from module.env_stocktrading import StockTradingEnv
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import datetime
import threading
from module.processor_alpaca import AlpacaProcessor
from module.config_tickers import DOW_30_TICKER
from module.config import (
    ALPACA_API_BASE_URL,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    INDICATORS,
    TODAY
)

class Alpaca():

    def __init__(self, model, turbulence_thresh=70):
        self.model = model
        self.turbulence_thresh = turbulence_thresh
        self.time_interval = 86400
        try:
            self.alpaca = tradeapi.REST(ALPACA_API_KEY,ALPACA_API_SECRET,ALPACA_API_BASE_URL, 'v2')
        except:
            raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')
        
        l = []
        for i in DOW_30_TICKER[:-1]:
            try:
                l.append(int(self.alpaca.get_position(i).qty))
            except:
                l.append(0)
                pass
        self.stocks = l #stocks holding
        l = []
        self.stocks_cd = np.zeros_like(self.stocks) 
        self.cash = float(self.alpaca.get_account().cash) #cash record 
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = DOW_30_TICKER[:-1])
        self.asset_list = []
        for i in DOW_30_TICKER[:-1]:
            try:
                l.append(float(self.alpaca.get_latest_bar(i).c))
            except:
                l.append(0)
                pass
        self.price = l
        l = []
        self.stockUniverse = DOW_30_TICKER[:-1]
        self.turbulence_bool = 0
        self.equities = []
        self.max_stocks = int(0.0001*(float(self.alpaca.get_account().cash)))


    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
          self.alpaca.cancel_order(order.id)
    
        # Wait for market to open.
        print("Waiting for market to open...")
        #tAMO = threading.Thread(target=self.awaitMarketOpen)
        #tAMO.start()
        #tAMO.join()
        print("Market opened.")
         # Figure out when the market will close so we can prepare to sell beforehand.
        clock = self.alpaca.get_clock()
        closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
        currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
        self.timeToClose = closingTime - currTime
    
        while True:
          
          if(self.timeToClose < (60)):
            # Close all positions when 1 minutes til market close.
            print("Market closing soon. Stop trading.")
            break

          else:
            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time,last_equity])
            time.sleep(self.time_interval)

    
    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open

        while(not isOpen):
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            print(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open


    def submitOrder(self, qty, stock, side, resp):
        if(qty > 0):
          try:
            self.alpaca.submit_order(stock, qty, side, "market", "day")
            print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
            resp.append(True)
          except:
            print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
            resp.append(False)
        else:
          print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
          resp.append(True)
    
    def trade(self):
        state = self.get_state()

        action = self.model.predict(state)[0]
        action = (action * self.max_stocks).astype(int)
        
        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = int(self.max_stocks*0.7)  # stock_cd
            sell = []
            buy = []

            for j in range(len(action[0])):
                if action[0][j] < -min_action:
                    sell.append(j)
                elif action[0][j] > min_action:
                    buy.append(j)

            for index in sell:  # sell_index:
                if self.stocks[index] > 0:   
                    sell_num_shares = abs(self.max_stocks//action[0][index])*100 
                    qty =  abs((int(sell_num_shares*0.01*self.stocks[index]))//2)
                else: 
                    sell_num_shares = abs(action[0][index])
                    qty = abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0
                try:    
                    self.stocks[index] = int(self.alpaca.get_position(self.stockUniverse[index]).qty)
                except:
                    self.stocks[index] = 0
                    pass

            for index in buy:  # buy_index:
                if self.cash < 25000:
                    print("Dodge PDT")
                    break
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(abs(tmp_cash // self.price[index]), abs((self.max_stocks // action[0][index]))*100)
                if (buy_num_shares != buy_num_shares): # if buy_num_change = nan
                    qty = 0 # set to 0 quantity
                elif self.stocks[index] < 0:
                    qty = abs(int(buy_num_shares*0.01*self.stocks[index]))
                else:
                    buy_num_shares = min(abs(tmp_cash // self.price[index]), abs(action[0][index]))
                    qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0
                try:    
                    self.stocks[index] = int(self.alpaca.get_position(self.stockUniverse[index]).qty)
                except:
                    self.stocks[index] = 0
                    pass
                
                
        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if(position.side == 'long'):
                    orderSide = 'sell'
                else:
                    orderSide = 'buy'
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
            
            self.stocks_cd[:] = 0   
            self.stocks = np.zeros(len(self.stockUniverse)) 
            
       
    def get_state(self):
        stock_dimension = len(DOW_30_TICKER[:-1])
        state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension

        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
        "hmax": int(0.0001*(float(self.alpaca.get_account().cash))),
        "initial_amount": float(self.alpaca.get_account().cash),
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
          }

        processor = AlpacaProcessor(API_SECRET= ALPACA_API_SECRET, API_BASE_URL= ALPACA_API_BASE_URL, API_KEY=ALPACA_API_KEY)
        d = pd.DataFrame(columns=['date', 'tic', 'open', 'high', 'low', 'close', 'volume', 'day', 'macd',
       'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
       'close_60_sma', 'vix', 'turbulence'])
        d["tic"] = DOW_30_TICKER
        d = d.drop(d.index[len(d)-1])

        data_df = pd.DataFrame()
        for tic in DOW_30_TICKER[:-1]:
            barset = self.alpaca.get_bars(tic, '1Min').df  # [tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = pd.concat([barset, data_df])
        data_df = data_df.reset_index()
        start_time = data_df['timestamp'][0]
        end_time = data_df['timestamp'][len(data_df)-1]

        for tic in range(len(DOW_30_TICKER)-1):
            tech, turb = processor.fetch_latest_data([DOW_30_TICKER[tic]], '1Min', INDICATORS, df = data_df, start_time = start_time, end_time = end_time)
            self.turbulence_bool = 1 if turb >= self.turbulence_thresh else 0
            turb = (self.sigmoid_sign(turb, self.turbulence_thresh) * 2 ** -5).astype(np.float32)
            
            dic = self.alpaca.get_latest_bar(DOW_30_TICKER[tic])
            dic = eval('{' + str(dic)[10:-2].replace('\n   ', '') + '}')
            d.iloc[tic,2] =np.float64(dic['o'])
            d.iloc[tic,3] = np.float64(dic['h'])
            d.iloc[tic,4] = np.float64(dic['l'])
            d.iloc[tic,5] = np.float64(dic['c'])
            d.iloc[tic,6] = np.float64(dic['v'])
            d.iloc[tic,-2] = np.float64(turb[0])
            d.iloc[tic,-1] = np.float64(0)
            d.iloc[tic, 0] = TODAY.strftime('%Y-%m-%d')
            d.iloc[tic, 7] = np.float64(datetime.datetime.weekday(TODAY))
            c = 0
            for i in range(len(INDICATORS)):
                d.iloc[tic,i+8] = np.float64(tech[c])
                c+=1
        d.index = d.date.factorize()[0]
        d = d.fillna(0)
        gym = StockTradingEnv(df = d, turbulence_threshold = self.turbulence_thresh,risk_indicator_col='vix', **env_kwargs)
        env, obs = gym.get_sb_env()
        return obs
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh



