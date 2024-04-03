import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import optuna

class TradingStrategyOptimizerCNN:
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
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_filters = trial.suggest_int('n_filters', 32, 128)
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        pool_size = trial.suggest_int('pool_size', 2, 3)
        strides = trial.suggest_int('strides', 1, 2)
        padding = trial.suggest_categorical('padding', ['valid', 'same'])

        # General neural network hyperparameters
        n_units = trial.suggest_int('n_units', 50, 200)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
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
        study.optimize(func, n_trials=10)
        return study.best_trial.params
    def build_and_train_cnn_model(self, best_params, X_train, y_train):
        n_layers = best_params['n_layers']
        n_filters = best_params['n_filters']
        kernel_size = best_params['kernel_size']
        pool_size = best_params.get('pool_size', 2)
        strides = best_params.get('strides', 1)
        padding = best_params.get('padding', 'valid')
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
