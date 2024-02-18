import math

import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

from sklearn import metrics

from models.DAGMM_module.dagmm import DAGMM

import warnings
warnings.filterwarnings('ignore')

class Dagmm1():
    def __init__(self, dataset) -> None:
        X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values
        self.X_train = []
        normal_data = X[self.y == 0]
        train_threshold = math.floor(0.7 * len(dataset))

        for i in range(train_threshold):
            self.X_train.append(normal_data[i])
        self.X_test = X

    def train_test(self):
        X_train = pd.DataFrame(self.X_train)
        model = DAGMM(
            comp_hiddens=[60, 30, 10, 1],
            comp_activation=tf.nn.tanh,
            est_hiddens=[10, 4],
            est_activation=tf.nn.tanh,
            est_dropout_ratio=0.5,
            learning_rate=0.0001,
            epoch_size=200,
            minibatch_size=1024,
            random_seed=1111 
        )
        model.fit(X_train)
        y_pred = model.predict(self.X_test)

        anomaly_energy_threshold = np.percentile(y_pred, 90)
        y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)

        mean_mse = np.mean(y_pred)
        std_mse = np.std(y_pred)
        high_t = mean_mse + 2 * std_mse
        low_t = mean_mse - 2 * std_mse

        precision_dagmm, recall_dagmm, f1_score_dagmm, _ = metrics.precision_recall_fscore_support(self.y, y_pred_flag, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y, y_pred)
        auc_roc_dagmm = metrics.auc(fpr, tpr)

        print("Classification Report (DAGMM): ")
        print('Precision: {:.4f}'.format(precision_dagmm))
        print('Recall: {:.4f}'.format(recall_dagmm))
        print('F1 score: {:.4f}'.format(f1_score_dagmm))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_dagmm))

        dagmm_res = {
            'y_true' : self.y,
            'y_pred' : y_pred_flag,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : round(auc_roc_dagmm, 4),
            'precision' : round(precision_dagmm, 4),
            'recall' : round(recall_dagmm, 4),
            'f1_score' : round(f1_score_dagmm, 4),
        }

        return dagmm_res