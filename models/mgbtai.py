import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans

import math
import warnings
warnings.simplefilter('ignore')

class MGBTAI():
    def __init__(self, df) -> None:
        self.leaf_nodes = []
        self.child_tree = []
        self.leaf_level_threshold = 4

        self.X = df.iloc[:, :-1]
        self.X = np.array(self.X)
        self.y_true = df.iloc[:, -1]

    def binary_tree(self, result, final_cluster, count, tree_dic):
        first_cluster = []
        second_cluster = []
        if count == 0:
            tree_dic[count] = result
        else:
            k = max(tree_dic) + 1
            tree_dic[k] = result
        
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(result)
        score = 0.1

        if score < 0.9 or count == 0:
            for i in range(len(result)):
                if final_cluster[i] == 1:
                    first_cluster.append(result[i])
                else:
                    second_cluster.append(result[i])
            count = count + 1

            if len(first_cluster) > self.min_cluster_threshold:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(first_cluster)
                self.binary_tree(first_cluster, kmeans.labels_, count, tree_dic)
            else:
                k = max(tree_dic) + 1
                tree_dic[k] = first_cluster
                self.leaf_nodes.append(k)

                if count <= self.leaf_level_threshold:
                    self.child_tree.extend(first_cluster)
            
            if len(second_cluster) > self.min_cluster_threshold:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(second_cluster)
                self.binary_tree(second_cluster, kmeans.labels_, count, tree_dic)
            else:
                k = max(tree_dic) + 1
                tree_dic[k] = second_cluster
                self.leaf_nodes.append(k)

                if count <= self.leaf_level_threshold:
                    self.child_tree.extend(second_cluster)
        
        else:
            k = max(tree_dic)
            self.leaf_nodes.append(k)
    
    def train(self):
        count = 0
        tree_dic = {}
        data_size = len(self.X)

        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(self.X)
        self.min_cluster_threshold = math.floor(0.2 * data_size)

        final_cluster = kmeans.labels_
        self.binary_tree(self.X, final_cluster, count, tree_dic)
        min_key = math.inf
        tot_cluster_len = 0

        for k in tree_dic:
            if k in self.leaf_nodes:
                tot_cluster_len = tot_cluster_len + len(tree_dic[k])
                if k < min_key:
                    min_key = k

        thresholds = [0.15, 0.1, 0.05, 0.2]
        curr_itr = 0

        while len(self.child_tree) > 0.2 * data_size:
            count = 0
            self.leaf_level_threshold = 3
            self.tree_dic = {}
            self.leaf_nodes = []
            child_tree_copy = self.child_tree
            self.child_tree = []

            if curr_itr < len(thresholds):
                self.min_cluster_threshold = math.floor(thresholds[curr_itr] * len(child_tree_copy))
            else:
                self.min_cluster_threshold = math.floor(thresholds[-1] * len(child_tree_copy))
            
            kmeans = KMeans(n_clusters=2, random_state=0).fit(child_tree_copy)
            final_cluster = kmeans.labels_
            self.binary_tree(child_tree_copy, final_cluster, count, tree_dic)

            min_key = 0
            tot_cluster_len = 0
            min_arr = tree_dic[min_key]

            for k in tree_dic:
                if k in self.leaf_nodes:
                    tot_cluster_len = tot_cluster_len + len(tree_dic[k])
                if len(tree_dic[k]) < len(min_arr):
                    min_key = k
                    min_arr = tree_dic[k]
            
            curr_itr = curr_itr + 1

    def test(self):
        childtree = np.vstack(self.child_tree)
        ind = []
        count = 0
        st_ts = self.X.tolist()
        st_ts = sorted(st_ts)
        st_ts = np.array(st_ts)
        st_ft = childtree.tolist()
        st_ft = sorted(st_ft)
        st_ft = np.array(st_ft)

        for i, e in enumerate(st_ts):
            if ((count <= (len(st_ft) - 1)) and np.array_equal(st_ft[count], e)):
                ind.append(i)
                while i + 1 < len(st_ts) and np.array_equal(st_ts[i], st_ts[i + 1]):
                    ind.append(i + 1)
                    i = i + 1
                count = count + 1

        print('Number of Anomalies: ', count)
        y_pred = np.zeros(len(self.X), dtype=int)
        y_pred[ind] = 1

        precision_mgbtai, recall_mgbtai, f1_score_mgbtai, _ = metrics.precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_mgbtai = metrics.auc(fpr, tpr)

        print("Classification Report (MGBTAI): ")
        print('Precision: {:.4f}'.format(precision_mgbtai))
        print('Recall: {:.4f}'.format(recall_mgbtai))
        print('F1 score: {:.4f}'.format(f1_score_mgbtai))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_mgbtai))

        mgbtai_res = {
            'y_true' : self.y_true,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : auc_roc_mgbtai,
            'precision' : precision_mgbtai,
            'recall' : recall_mgbtai,
            'f1_score' : f1_score_mgbtai,
        }

        return mgbtai_res