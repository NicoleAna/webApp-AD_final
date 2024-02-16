import math

import numpy as np

from sklearn import metrics

import tensorflow as tf

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import L2

import warnings
warnings.filterwarnings('ignore')

class AutoEncoder():
    def __init__(self, dataset) -> None:
        self.X_data = dataset.iloc[:, :-1].values
        self.y_true = dataset.iloc[:, -1].values
        tmp = len(self.X_data)
        df = self.X_data.astype('float32')
        df = np.array(df)
        normal_data = self.X_data[self.y_true == 0]
        self.train_size = math.floor(tmp * 0.7)
        self.test_data = self.X_data
        self.train_data = []

        for i in range(self.train_size):
            if i in range(len(normal_data)):
                self.train_data.append(normal_data[i])
        self.train_data = np.array(self.train_data)

        X = np.array(self.X_data)
        min_val = tf.reduce_min(X)
        max_val = tf.reduce_max(X)
        X = (X - min_val) / (max_val - min_val)
        X = tf.cast(X, tf.float32)

        self.ip_shape = X.shape[1]
        self.encoding_dim = 14
        self.hid_dim1 = int(self.encoding_dim/2)
        self.hid_dim2 = 4
        self.lr = 1e-7

        # build and compile autoencoder model
        self.autoencoder = self.build_model()
        self.autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

    # defining autoencoder
    def build_model(self):
        # encoder
        ip_layer = Input(shape=(self.ip_shape,))
        encoder = Dense(self.encoding_dim, activation='tanh', activity_regularizer=L2(self.lr))(ip_layer)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(self.hid_dim1, activation='relu')(encoder)
        encoder = Dense(self.hid_dim2, activation=tf.nn.leaky_relu)(encoder)

        # decoder
        decoder = Dense(self.hid_dim1, activation='relu')(encoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(self.encoding_dim, activation='relu')(decoder)
        decoder = Dense(self.ip_shape, activation='tanh')(decoder)

        autoencoder = Model(ip_layer, decoder)

        self.early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        return autoencoder
    
    def train_test(self, epochs=50, batch_size=64):
        history = self.autoencoder.fit(
            self.train_data, self.train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(self.test_data, self.test_data),
            verbose=1,
            callbacks=[self.early_stop]
        ).history

        preds = self.autoencoder.predict(self.test_data)
        mse = np.mean(np.power(self.test_data - preds, 2), axis=1)

        y_probs = mse
        lper = np.percentile(mse, 5)
        uper = np.percentile(mse, 95)

        y_pred = (mse <= lper) | (mse >= uper)

        precision_auto, recall_auto, f1_score_auto, _ = metrics.precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_auto = metrics.auc(fpr, tpr)

        print("Classification Report (AutoEncoders): ")
        print('Precision: {:.4f}'.format(precision_auto))
        print('Recall: {:.4f}'.format(recall_auto))
        print('F1 score: {:.4f}'.format(f1_score_auto))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_auto))

        auto_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : auc_roc_auto,
            'precision' : precision_auto,
            'recall' : recall_auto,
            'f1_score' : f1_score_auto,
        }

        return auto_res 