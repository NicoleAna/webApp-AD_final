from pyod.models.lof import LOF

import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

class Lof():
    def __init__(self, dataset) -> None:
        X_data = dataset.iloc[:, :-1]
        self.y_true = dataset.iloc[:, -1]
        df = X_data.astype('float32')
        df = np.array(df)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, self.y_true, test_size=0.3, random_state=42)
        self.X_test = X_data

    def train_test(self):
        clf = LOF()
        clf.fit(self.X_train)
        y_pred = clf.fit_predict(self.X_test)

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