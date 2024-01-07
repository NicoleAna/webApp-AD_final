import numpy as np

from sklearn import metrics

import tensorflow as tf
from keras.layers import Dense, LSTM, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam

import math
import warnings
warnings.filterwarnings('ignore')

class Lstm():
    def __init__(self, df) -> None:
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        self.X_data = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
        self.n_steps = 3
        self.y_true = self.y[self.n_steps:]
        train_threshold = math.floor(0.7 * len(df))

        self.X_train, self.y_train = self.split_sequence(self.X_data[:train_threshold][self.y[:train_threshold] == 0])
        self.X_test, self.y_test = self.split_sequence(self.X_data)
        # self.y_train.reshape(-1, 1)

        X = np.array(self.X_data)
        min_val = tf.reduce_min(X)
        max_val = tf.reduce_max(X)
        X = (X - min_val) / (max_val - min_val)
        X = tf.cast(X, tf.float32)

        self.ip_dim = X.shape[1]
        self.lstm_units = 14

        self.lstm = self.build_model()
        self.lstm.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=Adam(1e-7))

    def split_sequence(self, sequence):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + self.n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def build_model(self):
        ip_layer = Input(shape=(None, self.ip_dim))
        lstm_layer = LSTM(units=self.lstm_units, activation='tanh', return_sequences=True)(ip_layer)
        lstm_layer = Dropout(0.2)(lstm_layer)
        lstm_layer = LSTM(units=8, activation='tanh')(lstm_layer)
        op_layer = Dense(self.y_train.shape[1], activation='tanh')(lstm_layer)

        lstm_model = Model(ip_layer, op_layer)

        self.early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        return lstm_model
    
    def train_test(self, epochs, batch_size):
        history = self.lstm.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
            callbacks=[self.early_stop]
        ).history

        x_test_preds = self.lstm.predict(self.X_test)
        num_features_to_match = x_test_preds.shape[1]
        X_test_flattened = self.X_test.reshape(self.X_test.shape[0], -1) 
        X_test_flattened_subset = X_test_flattened[:, :num_features_to_match]

        mse = np.mean(np.power(X_test_flattened_subset - x_test_preds, 2), axis=1)
        mean_mse = np.mean(mse)
        std_mse = np.std(mse)
        lper = np.percentile(mse, 5)
        uper = np.percentile(mse, 95)

        y_pred = mse.copy()
        y_pred = np.array(y_pred)

        c = 0
        for i in range(len(y_pred)):
            if(mse[i] <= lper or mse[i] >= uper):
                c = c + 1
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        precision_lstm, recall_lstm, f1_score_lstm, _ = metrics.precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_lstm = metrics.auc(fpr, tpr)

        print("Classification Report (LSTM): ")
        print('Precision: {:.4f}'.format(precision_lstm))
        print('Recall: {:.4f}'.format(recall_lstm))
        print('F1 score: {:.4f}'.format(f1_score_lstm))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_lstm))

        lstm_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : auc_roc_lstm,
            'precision' : precision_lstm,
            'recall' : recall_lstm,
            'f1_score' : f1_score_lstm,
        }

        return lstm_res