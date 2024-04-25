import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans

from kneed import KneeLocator

import math
import warnings
warnings.filterwarnings('ignore')

class DBTAI():
    def __init__(self, df) -> None:
        self.leaf_nodes = []
        self.child_tree = []
        
        self.df = df

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

    def get_large_cluster_centroid(self, small_cluster_threshold, tree_dic):
        merged_clusters = {}
        i = 0
        for k in tree_dic:
            if k in self.leaf_nodes:
                if len(tree_dic[k]) > small_cluster_threshold:
                    merged_clusters[i] = tree_dic[k]
                    i = i + 1
        
        final_arr = []
        count = 0
        for k in merged_clusters:
            for j in range(len(merged_clusters[k])):
                arr = merged_clusters[k]
                final_arr.insert(count, arr[j])
                count = count + 1

        centroid = np.mean(final_arr, axis=0)
        return centroid
    
    def get_large_cluster_center(self, small_cluster_threshold, tree_dic):
        merged_clusters = {}
        centroid_arr = []
        i = 0
        for k in tree_dic:
            if k in self.leaf_nodes:
                if len(tree_dic[k]) > small_cluster_threshold:
                    merged_clusters[i] = tree_dic[k]
                    i = i + 1

        count = 0
        for k in merged_clusters:
            centroid = np.mean(merged_clusters[k], axis=0)
            centroid_arr.insert(count, centroid)
            count = count + 1

        return centroid_arr
    
    def get_large_cluster_anomaly_score(self, small_cluster_threshold, tree_dic):
        large_clusters = {}
        ano_large_clusters = {}
        for k in tree_dic:
            if k in self.leaf_nodes:
                if len(tree_dic[k]) >= small_cluster_threshold:
                    large_clusters[k] = tree_dic[k]

        for k in large_clusters:
            arr = large_clusters[k]
            centroid = np.mean(arr, axis=0)
            for i in range(len(large_clusters[k])):
                ano_score_large_cluster = np.linalg.norm(arr[i] - centroid)
                ano_large_clusters[ano_score_large_cluster] = arr[i]

        return ano_large_clusters

    def train(self):
        count = 0
        tree_dic = {}
        df = self.df
        df = np.array(df)
        st_ts = df.tolist()
        st_ts = sorted(st_ts)
        st_ts = np.array(st_ts)
        self.actual = st_ts[:, -1]
        self.ts = self.df.iloc[:, :-1]
        self.ts = np.array(self.ts)
        data_size = len(self.ts)

        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(self.ts)
        small_cluster_threshold = math.floor(0.02 * data_size)
        self.leaf_level_threshold = 3
        self.min_cluster_threshold = math.floor(0.1 * data_size)

        final_cluster = kmeans.labels_
        self.binary_tree(self.ts, final_cluster, count, tree_dic)
        min_key = math.inf
        tot_cluster_len = 0

        for k in tree_dic:
            if k in self.leaf_nodes:
                tot_cluster_len = tot_cluster_len + len(tree_dic[k])
                if k < min_key:
                    min_key = k

        anomaly_score_dict = {}
        anomalu_score_from_large_clusters = {}
        merged_clusters = {}
        merged_itr = 0

        for k in tree_dic:
            if len(tree_dic[k]) < small_cluster_threshold:
                merged_clusters[merged_itr] = tree_dic[k]
                merged_itr = merged_itr + 1

        final_arr = []

        for k in merged_clusters:
            for i in range(len(merged_clusters[k])):
                arr = merged_clusters[k]
                final_arr.insert(count, arr[i])
                count = count + 1

        centroid = np.mean(final_arr, axis=0)
        
        # anomaly score by subtracting value from centroid
        for i in range(len(final_arr)):
            ano_score = np.linalg.norm(final_arr[i] - centroid)
            anomaly_score_dict[ano_score] = final_arr[i]

        normal_cluster_centroid = self.get_large_cluster_centroid(small_cluster_threshold, tree_dic)
        for i in range(len(final_arr)):
            ano_score_large_cluster = np.linalg.norm(final_arr[i] - normal_cluster_centroid)
            anomalu_score_from_large_clusters[ano_score_large_cluster] = final_arr[i]

        centroid_arr = self.get_large_cluster_center(small_cluster_threshold, tree_dic)

        # get cblof : cluster based local outlier factor, which calculates the min distance of the data points belonging to the smallest clusters to the largest clusters centriod 
        cblof = {}
        for i in range(len(final_arr)):
            mini = math.inf
            for c in range(len(centroid_arr)):
                cblof_dist = np.linalg.norm(final_arr[i] - centroid_arr[c])
                if cblof_dist < mini:
                    mini = cblof_dist
                if c == len(centroid_arr) - 1:
                    cblof[mini] = final_arr[i]

        cblof_large_cluster = self.get_large_cluster_anomaly_score(small_cluster_threshold, tree_dic)
        final_merged_cblof = {**cblof, **cblof_large_cluster}
        ano_score_list = list(final_merged_cblof.keys())

        sorted_scores = np.sort(ano_score_list)
        cumulative_sum = np.cumsum(sorted_scores)
        per_sum = (cumulative_sum / cumulative_sum[-1]) * 100

        knee = KneeLocator(range(1, len(per_sum) + 1), per_sum, curve='convex', direction='increasing')
        knee_threshold = sorted_scores[knee.elbow]

        self.y_anom = [d for ano_score, d in final_merged_cblof.items() if ano_score > knee_threshold]

        print('Total number of anomalies: ', len(self.y_anom))

    def test(self):
        ind = []
        count = 0
        y_anom1 = np.vstack(self.y_anom)
        st_ts = self.ts.tolist()
        st_ts = sorted(st_ts)
        st_ts = np.array(st_ts)
        st_ft = y_anom1.tolist()
        st_ft = sorted(st_ft)
        st_ft = np.array(st_ft)

        for i, e in enumerate(st_ts):
            if ((count <= (len(st_ft) - 1)) and np.allclose(st_ft[count], e, atol=1e-04)):
                ind.append(i)
                while i + 1 < len(st_ts) and np.allclose(st_ts[i], st_ts[i + 1], atol=1e-04):
                    ind.append(i + 1)
                    i = i + 1
                count = count + 1

        y_pred = np.zeros(len(self.ts), dtype=int)
        y_pred[ind] = 1

        precision_dbtai, recall_dbtai, f1_score_dbtai, _ = metrics.precision_recall_fscore_support(self.actual, y_pred, average='binary')
        fpr, tpr, _ = metrics.roc_curve(self.actual, y_pred)
        auc_roc_dbtai = metrics.auc(fpr, tpr)

        print("Classification Report (DBTAI): ")
        print('Precision: {:.4f}'.format(precision_dbtai))
        print('Recall: {:.4f}'.format(recall_dbtai))
        print('F1 score: {:.4f}'.format(f1_score_dbtai))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_dbtai))

        dbtai_res = {
            'y_true' : self.actual,
            'y_pred' : y_pred,
            'fpr' : fpr,
            'tpr' : tpr,
            'auc_roc' : round(auc_roc_dbtai, 4),
            'precision' : round(precision_dbtai, 4),
            'recall' : round(recall_dbtai, 4),
            'f1_score' : round(f1_score_dbtai, 4),
        }

        return dbtai_res