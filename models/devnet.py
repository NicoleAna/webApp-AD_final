from deepod.models.devnet import DevNet

import numpy as np

from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

class Devnet():
    def __init__(self, dataset) -> None:
        train_feat = dataset.iloc[:, :-1].values
        train_lab = dataset.iloc[:, -1].values
        normal_data = train_feat[train_lab == 0]
        anomaly_data = train_feat[train_lab == 1]
        tot_samples = len(train_feat)
        num_samples_train = int(0.7 * tot_samples)
        num_samples_anomaly = int(0.02 * num_samples_train)

        np.random.shuffle(normal_data)

        self.X_train = np.vstack((normal_data[:num_samples_train - num_samples_anomaly], anomaly_data[:num_samples_anomaly]))
        self.y_train = np.hstack((np.zeros(len(normal_data[:num_samples_train - num_samples_anomaly])), np.ones(len(anomaly_data[:num_samples_anomaly]))))

        self.test_feat = dataset.iloc[:, :-1].values
        self.test_lab = dataset.iloc[:, -1].values

    def train_test(self):
        clf = DevNet(epochs=10)
        clf.fit(self.X_train, self.y_train)

        preds = clf.predict(self.test_feat)
        scores = clf.decision_function(self.test_feat)
        print(scores)

        precision_dev, recall_dev, f1_score_dev, _ = metrics.precision_recall_fscore_support(self.test_lab, preds, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.test_lab, preds)
        auc_roc_dev = metrics.auc(fpr, tpr)

        print("Classification Report (Deviation Net): ")
        print('Precision: {:.4f}'.format(precision_dev))
        print('Recall: {:.4f}'.format(recall_dev))
        print('F1 score: {:.4f}'.format(f1_score_dev))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_dev))

        dev_res = {
            'y_true' : self.test_lab,
            'y_pred' : preds,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : auc_roc_dev,
            'precision' : precision_dev,
            'recall' : recall_dev,
            'f1_score' : f1_score_dev,
        }

        return dev_res