import numpy as np 

from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

class ellipticEnvelope():
    def __init__(self, dataset) -> None:
        self.X_data = dataset.iloc[:, :-1]
        self.y_true = dataset.iloc[:, -1]
        df = self.X_data.astype('float32')
        df = np.array(df)
        self.train_data, _, _, _ = train_test_split(self.X_data, self.y_true, test_size=0.3, random_state=42)
        self.test_data = df

    def train_test(self):
        clf = EllipticEnvelope(random_state=42)
        clf.fit(self.train_data)
        y_pred = clf.predict(self.test_data)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        precision_env = metrics.precision_score(self.y_true, y_pred)
        recall_env = metrics.recall_score(self.y_true, y_pred)
        f1_score_env = metrics.f1_score(self.y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_env = metrics.auc(fpr, tpr)

        print("Classification Report (Elliptic Envelope): ")
        print('Precision: {:.4f}'.format(precision_env))
        print('Recall: {:.4f}'.format(recall_env))
        print('F1 score: {:.4f}'.format(f1_score_env))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_env))

        env_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : auc_roc_env,
            'precision' : precision_env,
            'recall' : recall_env,
            'f1_score' : f1_score_env,
        }

        return env_res