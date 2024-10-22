import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

class iForest():
    def __init__(self, dataset) -> None:
        self.X_data = dataset.iloc[:, :-1]
        self.y_true = dataset.iloc[:, -1]
        df = self.X_data.astype('float32')
        df = np.array(df)
        self.train_data, _, _, _ = train_test_split(self.X_data, self.y_true, test_size=0.3, random_state=42)
        self.test_data = df

    def train_test(self):
        clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12),
                              bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
        clf.fit(self.train_data)
        y_pred = clf.predict(self.test_data)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        precision_iforest = metrics.precision_score(self.y_true, y_pred)
        recall_iforest = metrics.recall_score(self.y_true, y_pred)
        f1_score_iforest = metrics.f1_score(self.y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_iforest = metrics.auc(fpr, tpr)

        print("Classification Report (IForest): ")
        print('Precision: {:.4f}'.format(precision_iforest))
        print('Recall: {:.4f}'.format(recall_iforest))
        print('F1 score: {:.4f}'.format(f1_score_iforest))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_iforest))

        iforest_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : round(auc_roc_iforest, 4),
            'precision' : round(precision_iforest, 4),
            'recall' : round(recall_iforest, 4),
            'f1_score' : round(f1_score_iforest, 4),
        }

        return iforest_res