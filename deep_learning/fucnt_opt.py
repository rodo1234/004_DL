import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


import tensorflow as tf

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares,
                 stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

class TradingStrategyOptimizer:
    def __init__(self, buy_data_path, sell_data_path):
        self.buy_data_path = buy_data_path
        self.sell_data_path = sell_data_path
        
    
    def load_and_preprocess_data(self, path):
        data = pd.read_csv(path)
        y = data.pop('Y_BUY' if 'buy' in path else 'Y_SELL').values
        X = data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', 50, 200)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_units, input_dim=X_train.shape[1], activation='relu'))
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(n_units, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=False)
        _, accuracy = model.evaluate(X_val, y_val, verbose=False)
        return accuracy
    
    def optimize_dnn(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction='maximize')
        func = lambda trial: self.objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(func, n_trials=10)
        return study.best_trial.params

    def build_and_train_model(self, best_params, X_train, y_train):
        n_layers = best_params['n_layers']
        n_units = best_params['n_units']
        lr = best_params['lr']
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_units, input_dim=X_train.shape[1], activation='relu'))
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(n_units, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, verbose=False)
        return model
    
    def run_strategy(self):
        for i, row in self.df.iterrows():

            # Close Operations
            temp_operations = []
            for op in self.active_operations:
                if op.operation_type == 'Long':
                    if op.stop_loss > row.Close:  # Close losing operations
                        self.cash += row.Close * op.n_shares * (1 - self.com)
                    elif op.take_profit < row.Close:  # Close profits
                        self.cash += row.Close * op.n_shares * (1 - self.com)
                    else:
                        temp_operations.append(op)
                elif op.operation_type == 'Short':
                    if op.stop_loss < row.Close:  # Close losing operations
                        self.cash -= row.Close * op.n_shares * (1 + self.com)
                    elif op.take_profit > row.Close:  # Close profits
                        self.cash -= row.Close * op.n_shares * (1 + self.com)
                    else:
                        temp_operations.append(op)
            self.active_operations = temp_operations
            
            # Open Operations
            if row.Y_BUY:
                n_shares = self.n_shares_long
                stop_loss = row.Close * (1 - self.stop_loss_long)
                take_profit = row.Close * (1 + self.take_profit_long)
                self.active_operations.append(Operation('Long', row.Close, row.timestamp, n_shares, stop_loss, take_profit))
                self.cash -= row.Close * n_shares * (1 + self.com)
            elif row.Y_SELL:
                n_shares = self.n_shares_short
                stop_loss = row.Close * (1 + self.stop_loss_short)
                take_profit = row.Close * (1 - self.take_profit_short)
                self.active_operations.append(Operation('Short', row.Close, row.timestamp, n_shares, stop_loss, take_profit))
                self.cash += row.Close * n_shares * (1 - self.com)
                
            self.strategy_value.append(self.cash)
            
            return self.strategy_value[-1]
    

# Uso de la clase TradingStrategyOptimizer
buy_data_path = 'data/close_data_buy_5.csv'
sell_data_path = 'data/close_data_sell_5.csv'
# Parámetros óptimos
stop_loss_long = 0.0317407648006915
take_profit_long = 0.04998058375999087
stop_loss_short = 0.049689221836262565
take_profit_short = 0.021530456551386253
n_shares_long = 97
n_shares_short = 99

# Variables de la estrategia
cash = 1_000_000
active_operations = []
com = 0.00125
strategy_value = [1_000_000]
optimizer = TradingStrategyOptimizer(buy_data_path, sell_data_path)
optimizer.run_strategy()

    
