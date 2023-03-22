from module.models import DRLAgent
from module.env_stocktrading import StockTradingEnv
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import datetime
import threading

from finrl.meta.data_processor import Alpaca
from module.config_tickers import DOW_30_TICKER

from module.config import (
    ALPACA_API_BASE_URL,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    INDICATORS
)

class Alpaca():

    def __init__(self, model, turbulence_thresh=70):
        self.model = model
        self.turbulence_thresh = turbulence_thresh

        try:
            self.alpaca = tradeapi.REST(ALPACA_API_KEY,ALPACA_API_SECRET,ALPACA_API_BASE_URL, 'v2')
        except:
            raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')
        
        self.stocks = np.asarray([0] * len(DOW_30_TICKER)) #stocks holding
        self.stocks_cd = np.zeros_like(self.stocks) 
        self.cash = None #cash record 
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = DOW_30_TICKER)
        self.asset_list = []
        self.price = np.asarray([0] * len(DOW_30_TICKER))
        self.stockUniverse = DOW_30_TICKER
        self.turbulence_bool = 0
        self.equities = []

    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
          self.alpaca.cancel_order(order.id)
    
        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
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

        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty =  abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
                if (buy_num_shares != buy_num_shares): # if buy_num_change = nan
                    qty = 0 # set to 0 quantity
                else:
                    qty = abs(int(buy_num_shares))
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0
                
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

           
    def get_state(self):
       alpaca = Alpaca(api=self.alpaca)
       price, tech, turbulence = alpaca.fetch_latest_data(ticker_list = DOW_30_TICKER, time_interval='1Min',
                                                     tech_indicator_list=INDICATORS)

       turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
        
       turbulence = (self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2 ** -5).astype(np.float32)
        
       tech = tech * 2 ** -7
       positions = self.alpaca.list_positions()
       stocks = [0] * len(DOW_30_TICKER)
       for position in positions:
          ind = DOW_30_TICKER.index(position.symbol)
          stocks[ind] = ( abs(int(float(position.qty))))
        
       stocks = np.asarray(stocks, dtype = float)
       cash = float(self.alpaca.get_account().cash)
       self.cash = cash
       self.stocks = stocks
       self.turbulence_bool = turbulence_bool 
       self.price = price
        
        
        
       amount = np.array(self.cash * (2 ** -12), dtype=np.float32)
       scale = np.array(2 ** -6, dtype=np.float32)
       state = np.hstack((amount,
                    turbulence,
                    self.turbulence_bool,
                    price * scale,
                    self.stocks * scale,
                    self.stocks_cd,
                    tech,
                    )).astype(np.float32)
       state[np.isnan(state)] = 0.0
       state[np.isinf(state)] = 0.0
       return state
    
    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh