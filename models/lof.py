from pyod.models.lof import LOF

import numpy as np

from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

class Lof():
    def __init__(self, dataset) -> None:
        self.X_data = dataset.iloc[:, :-1]
        self.y_true = dataset.iloc[:, -1]
        tmp = len(self.X_data)
        df = self.X_data.astype('float32')
        df = np.array(df)

        self.train_size = int(tmp * 0.7)
        self.test_data = df
        self.train_data = []

        for i in range(self.train_size):
            self.train_data.append(df[i])
        self.train_data = np.array(self.train_data)

    def train_test(self):
        clf = LOF()
        clf.fit(self.train_data)
        y_pred = clf.fit_predict(self.test_data)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        precision_lof = metrics.precision_score(self.y_true, y_pred)
        recall_lof = metrics.recall_score(self.y_true, y_pred)
        f1_score_lof = metrics.f1_score(self.y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_lof = metrics.auc(fpr, tpr)

        print("Classification Report (LOF): ")
        print('Precision: {:.4f}'.format(precision_lof))
        print('Recall: {:.4f}'.format(recall_lof))
        print('F1 score: {:.4f}'.format(f1_score_lof))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_lof))

        lof_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : round(auc_roc_lof, 4),
            'precision' : round(precision_lof, 4),
            'recall' : round(recall_lof, 4),
            'f1_score' : round(f1_score_lof, 4),
        }

        return lof_res