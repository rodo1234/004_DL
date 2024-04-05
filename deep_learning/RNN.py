import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TradingStrategyOptimizer:
    def _init_(self, buy_data_path, sell_data_path):
        self.buy_data_path = buy_data_path
        self.sell_data_path = sell_data_path

    def load_and_preprocess_data(self, path):
        data = pd.read_csv(path)
        y = data.pop('Y_BUY' if 'buy' in path else 'Y_SELL').values
        X = data.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Suponiendo que los datos ya están en una forma que las RNN pueden procesar
        # Es posible que necesites remodelar X_scaled para que sea compatible con RNN
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        return train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    def objective(self, trial, X_train, y_train, X_val, y_val):
        n_units = trial.suggest_int('n_units', 20, 100)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.SimpleRNN(n_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=False)
        _, accuracy = model.evaluate(X_val, y_val, verbose=False)
        return accuracy

    def optimize_rnn(self, X_train, y_train, X_val, y_val):
        study = optuna.create_study(direction='maximize')
        func = lambda trial: self.objective(trial, X_train, y_train, X_val, y_val)
        study.optimize(func, n_trials=10)
        return study.best_trial.params

    def build_and_train_model(self, best_params, X_train, y_train):
        n_units = best_params['n_units']
        lr = best_params['lr']

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.SimpleRNN(n_units, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, verbose=False)
        return model