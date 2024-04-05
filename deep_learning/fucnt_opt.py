import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import ta

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
    
    def run(self):
        X_train_buy, X_val_buy, y_train_buy, y_val_buy = self.load_and_preprocess_data(self.buy_data_path)
        X_train_sell, X_val_sell, y_train_sell, y_val_sell = self.load_and_preprocess_data(self.sell_data_path)
        
        best_params_buy = self.optimize_dnn(X_train_buy, y_train_buy, X_val_buy, y_val_buy)
        best_params_sell = self.optimize_dnn(X_train_sell, y_train_sell, X_val_sell, y_val_sell)
        
        model_buy = self.build_and_train_model(best_params_buy, X_train_buy, y_train_buy)
        model_sell = self.build_and_train_model(best_params_sell, X_train_sell, y_train_sell)
        
        datos = pd.read_csv(self.buy_data_path)
        y = datos.pop('Y_BUY' if 'buy' in self.buy_data_path else 'Y_SELL').values
        X = datos.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        y_pred_buy = model_buy.predict(X_scaled)
        y_pred_sell = model_sell.predict(X_scaled)
        
        # covertir a booleanos
        y_pred_buy = y_pred_buy > 0.5
        y_pred_sell = y_pred_sell > 0.5
        
        
        return pd.DataFrame({'Y_BUY_PRED': y_pred_buy.flatten(), 'Y_SELL_PRED': y_pred_sell.flatten()})
    
# if __name__ == '__main__':
#     optimizer = TradingStrategyOptimizer('/home/rodo/code/proyecto_4/004_DL/data/close_data_buy_5.csv', '/home/rodo/code/proyecto_4/004_DL/data/close_data_sell_5.csv')
#     df = optimizer.run()
#     print(df)
    
def clean_ds(df):
    df = df.copy()
    for i in range(1, 6):
        df[f'X_t-{i}'] = df['Close'].shift(i)

    df['Pt_5'] = df['Close'].shift(-5)

    # Agregamos RSI
    rsi_data = ta.momentum.RSIIndicator(close=df['Close'], window=28)
    df['RSI'] = rsi_data.rsi()

    # La Y
    df['Y_BUY'] = df['Close'] < df['Pt_5']
    df['Y_SELL'] = df['Close'] > df['Pt_5']

    # df['Y_BUY'] = df['Y_BUY'].astype(int)
    # df['Y_SELL'] = df['Y_SELL'].astype(int)

    return df

# df1 = pd.read_csv("data/aapl_5m_train.csv")
# df_5min = clean_ds(df1)

# close_data = df_5min[['Timestamp','Close', 'X_t-1', 'X_t-2', 'X_t-3', 'X_t-4' ,'X_t-5','RSI', 'Y_BUY']]
# close_data = close_data.dropna()

# df_indexed = df.reset_index()
# close_data_indexed = close_data.reset_index()

# close_data_updated = close_data_indexed.join(df_indexed[['Y_BUY_PRED', 'Y_SELL_PRED']])



# closes_5min = close_data_updated[['Timestamp', 'Close','Y_BUY_PRED', 'Y_SELL_PRED']]

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares,stop_loss, take_profit):
        self.df = self.df
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        #self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
class dnn_strategy:
    def __init__(self, df, cash, active_operations, com, n_shares, stop_loss, take_profit):
        self.df = df
        self.cash = cash
        self.active_operations = active_operations
        self.com = com
        self.n_shares = n_shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_value = []
        
    def run_strategy(self):
        for i, row in self.df.iterrows():

            # Close Operations
            temp_operations = []
            for op in self.active_operations:
                if op.operation_type == 'Long':
                    if op.stop_loss > row.Close:  
                        self.cash += row.Close * op.n_shares * (1 - self.com)
                    elif op.take_profit < row.Close:  
                        self.cash += row.Close * op.n_shares * (1 - self.com)
                    else:
                        temp_operations.append(op)
                elif op.operation_type == 'Short':
                    if op.stop_loss < row.Close:  
                        self.cash -= row.Close * op.n_shares * (1 + self.com)
                    elif op.take_profit > row.Close:  
                        self.cash -= row.Close * op.n_shares * (1 + self.com)
                    else:
                        temp_operations.append(op)
            self.active_operations = temp_operations
            
            # Open Operations
            if row.Y_BUY_PRED:
                n_shares = self.n_shares
                stop_loss = row.Close * (1 - self.stop_loss)
                take_profit = row.Close * (1 + self.take_profit)
                self.active_operations.append(Operation(self.df,'Long', row.Close, row.Timestamp, n_shares, stop_loss, take_profit))
                self.cash += row.Close * n_shares * (1 + self.com)
            elif row.Y_SELL_PRED:
                n_shares = self.n_shares
                stop_loss = row.Close * (1 + self.stop_loss)
                take_profit = row.Close * (1 - self.take_profit)
                self.active_operations.append(Operation(self.df,'Short', row.Close, row.Timestamp, n_shares, stop_loss, take_profit))
                self.cash -= row.Close * n_shares * (1 - self.com)
                
            
            total_value = len(self.active_operations) * row.Close * self.n_shares
            self.strategy_value.append(self.cash + total_value)
            return self.strategy_value[-1] 
        

cash = 1_000_000
active_operations = []
com = 0.00125  # comision en GBM
strategy_value = [1_000_000]

best_global_strategy = {'name': None, 'value': float('-inf')}

def optimize(trial,):
    # Definición de los parámetros a optimizar
    stop_loss = trial.suggest_float('stop_loss', 0.00250, 0.05)
    take_profit = trial.suggest_float('take_profit', 0.00250, 0.05)
    n_shares = trial.suggest_int('n_shares', 5, 200)
    
    dnn_strat = dnn_strategy(
        df=df,  # df
        cash=cash,  # saldo inicial
        active_operations=[],
        com=com,  # comisión GBM
        n_shares=n_shares,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    dnn_strat.run_strategy()
    
    strategy_values = {
        'dnn_strategy': dnn_strat.run_strategy()
    }
    
    best_strategy_name = max(strategy_values, key=strategy_values.get)
    best_strategy_value = strategy_values[best_strategy_name]

    if best_strategy_value > best_global_strategy['value']:
        best_global_strategy['name'] = best_strategy_name
        best_global_strategy['value'] = best_strategy_value

    
    # Retorna el valor de la mejor estrategia
    return best_strategy_value

# Inicializar y ejecutar la optimización
# study = optuna.create_study(direction='maximize')
# study.optimize(optimize, n_trials=1000, n_jobs=-1)

# # Los mejores parámetros encontrados en el mejor trial
# best_params = study.best_trial.params
# best_value = study.best_trial.value

# # Comparar con el mejor valor global previamente encontrado y el nombre de la estrategia
# best_strategy_name = best_global_strategy['name']
# best_strategy_value = best_global_strategy['value']

# # Imprimir los resultados, incluido el nombre de la mejor estrategia global y su valor
# print(f"Best buy overall strategy: {best_strategy_name} with value: {best_strategy_value}")
# print("Best buy strategy parameters:", best_params)


# #plot the strategy value over time 
# plt.plot(strategy_value)
# plt.title('DNN Strategy Value Over Time')
# plt.xlabel('Time')
# plt.ylabel('Strategy Value')
# plt.show()



    
