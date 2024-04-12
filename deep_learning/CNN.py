import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import optuna
import ta

class TradingStrategyOptimizer_CNN:
    def __init__(self, buy_data_path, sell_data_path):
        self.buy_data_path = buy_data_path
        self.sell_data_path = sell_data_path

    def load_and_preprocess_data(self, path):
        data = pd.read_csv(path)
        target_column = 'Y_BUY' if 'buy' in path else 'Y_SELL'
        y = data.pop(target_column).astype(int).values
        X = data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Reshape for Conv1D: (samples, timesteps, features)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # CNN-specific hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 5)
        n_filters = trial.suggest_int('n_filters', 32, 128)
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        pool_size = trial.suggest_int('pool_size', 1, 2)
        strides = trial.suggest_int('strides', 1, 2)
        padding = trial.suggest_categorical('padding', ['valid', 'same'])

        # General neural network hyperparameters
        n_units = trial.suggest_int('n_units', 50, 200)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation,
                                         input_shape=(X_train.shape[1], X_train.shape[2]), padding=padding,
                                         strides=strides))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
        model.add(tf.keras.layers.Flatten())
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(n_units, activation=activation))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=False)
        _, accuracy = model.evaluate(X_val, y_val, verbose=False)
        return accuracy

    def optimize_cnn(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction='maximize')
        func = lambda trial: self.objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(func, n_trials=20)
        return study.best_trial.params
    def build_and_train_cnn_model(self, best_params, X_train, y_train):
        n_layers = best_params['n_layers']
        n_filters = best_params['n_filters']
        kernel_size = best_params['kernel_size']
        pool_size = best_params.get('pool_size')
        strides = best_params.get('strides')
        padding = best_params.get('padding')
        n_units = best_params['n_units']
        activation = best_params['activation']
        lr = best_params['lr']

        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, activation=activation,
                                   input_shape=(X_train.shape[1], X_train.shape[2]), padding=padding, strides=strides),
            tf.keras.layers.MaxPooling1D(pool_size=pool_size),
            tf.keras.layers.Flatten()
        ])
        for _ in range(n_layers - 1):
            model.add(tf.keras.layers.Dense(n_units, activation=activation))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, verbose=2)
        return model

    def run_cnn(self):
        X_train_buy, X_val_buy, y_train_buy, y_val_buy = self.load_and_preprocess_data(self.buy_data_path)
        X_train_sell, X_val_sell, y_train_sell, y_val_sell = self.load_and_preprocess_data(self.sell_data_path)

        best_params_buy = self.optimize_cnn(X_train_buy, y_train_buy, X_val_buy, y_val_buy)
        best_params_sell = self.optimize_cnn(X_train_sell, y_train_sell, X_val_sell, y_val_sell)

        model_buy_cnn = self.build_and_train_cnn_model(best_params_buy, X_train_buy, y_train_buy)
        model_sell_cnn = self.build_and_train_cnn_model(best_params_sell, X_train_sell, y_train_sell)

        datos = pd.read_csv(self.buy_data_path)
        y = datos.pop('Y_BUY' if 'buy' in self.buy_data_path else 'Y_SELL').values
        X = datos.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_pred_buy = model_buy_cnn.predict(X_scaled)
        y_pred_sell = model_sell_cnn.predict(X_scaled)
        
        #Guardar el modelo
        model_buy_cnn.save('model_buy_cnn.keras')
        model_sell_cnn.save('model_sell_cnn.keras')

        # covertir a booleanos
        y_pred_buy = y_pred_buy > 0.5
        y_pred_sell = y_pred_sell > 0.5

        return pd.DataFrame({'Y_BUY_PRED_CNN': y_pred_buy.flatten(), 'Y_SELL_PRED_CNN': y_pred_sell.flatten()}), model_buy_cnn, model_sell_cnn

def clean_ds_CNN(df):
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

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        # self.sold_at = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

class cnn_strategy:
    def __init__(self, df, cash, active_operations, com, n_shares, stop_loss, take_profit):
        self.df = df
        self.cash = cash
        self.active_operations = active_operations
        self.com = com
        self.n_shares = n_shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_value = []

    def run_strategy_cnn(self):
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
            if row.Y_BUY_PRED_CNN:
                n_shares = self.n_shares
                stop_loss = row.Close * (1 - self.stop_loss)
                take_profit = row.Close * (1 + self.take_profit)
                self.active_operations.append(
                    Operation('Long', row.Close, row.Timestamp, n_shares, stop_loss, take_profit))
                self.cash -= row.Close * n_shares * (1 + self.com)
            elif row.Y_SELL_PRED_CNN:
                n_shares = self.n_shares
                stop_loss = row.Close * (1 + self.stop_loss)
                take_profit = row.Close * (1 - self.take_profit)
                self.active_operations.append(
                    Operation('Short', row.Close, row.Timestamp, n_shares, stop_loss, take_profit))
                self.cash += row.Close * n_shares * (1 - self.com)

            total_value = len(self.active_operations) * row.Close 
            self.strategy_value.append(self.cash + total_value)
        return self.strategy_value[-1]
