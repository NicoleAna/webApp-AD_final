import numpy as np
import pandas as pd

from sklearn import metrics

import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
# from keras.optimizers import Adam
from keras import backend as K
tf.compat.v1.enable_eager_execution()

import math
import warnings
warnings.filterwarnings('ignore')

class QReg():
    def __init__(self, df) -> None:
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        self.n_steps = 3
        self.y_true = y[self.n_steps:]
        train_threshold = math.floor(0.7 * len(df))

        self.X_train, self.y_train = self.split_sequence(X[:train_threshold][y[:train_threshold] == 0])
        self.X_test, self.y_test = self.split_sequence(X)

        self.q_model = self.build_model()
        self.q_model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=self.QuantileLoss())

    def split_sequence(self, sequence):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + self.n_steps   # find the end of this pattern
            if end_ix > len(sequence)-1:    # check is we are beyond the seq
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]     # gather ip and op parts of the pattern
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def QuantileLoss(self):
        delta=1e-4
        perc_points = [0.01, 0.25, 0.5, 0.75, 0.99]
        perc = np.array(perc_points).reshape(-1)
        perc.sort()
        perc = perc.reshape(1, -1)

        def _qloss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            I = tf.cast(y_true <= y_pred, tf.float32)
            d = K.abs(y_true - y_pred)
            correction = I * (1 - perc) + (1 - I) * perc
            # huber loss
            huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
            # order loss
            q_order_loss = K.sum(K.maximum(0.0, y_pred[:, :-1] - y_pred[:, 1:] + 1e-6), -1)
            return huber_loss + q_order_loss
        return _qloss


    def build_model(self):
        model = Sequential(
            [
                LSTM(64, input_shape=(self.n_steps, 1), activation='sigmoid', recurrent_activation='sigmoid'),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(5, activation='sigmoid')
            ]
        )
        return model

    def train_test(self):
        self.q_model.fit(self.X_train, self.y_train, epochs=50, verbose=0)
        preds = self.q_model.predict(self.X_test)
        predictions = list()

        for i in range(len(preds)):
            inner_loop = preds[i]
            predictions.append(np.absolute(inner_loop[0] - inner_loop[4]))

        c = 0
        lper = np.percentile(predictions, 0.9)
        uper = np.percentile(predictions, 99.1)
        y_pred_algo = (predictions <= lper) | (predictions >= uper)

        y_pred = np.zeros(len(self.y_true))
        for i in range(len(y_pred)):
            if (predictions[i] <= lper or predictions[i] >= uper):
                c = c + 1
                y_pred[i] = 1
        print('Count of anomalies', c)

        precision_qreg, recall_qreg, f1_score_qreg, _ = metrics.precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_qreg = metrics.auc(fpr, tpr)

        print("Classification Report (Quantile Regression): ")
        print('Precision: {:.4f}'.format(precision_qreg))
        print('Recall: {:.4f}'.format(recall_qreg))
        print('F1 score: {:.4f}'.format(f1_score_qreg))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_qreg))

        qreg_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : round(auc_roc_qreg, 4),
            'precision' : round(precision_qreg, 4),
            'recall' : round(recall_qreg, 4),
            'f1_score' : round(f1_score_qreg, 4),
        }

        return qreg_res