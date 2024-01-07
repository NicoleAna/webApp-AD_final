import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from io import BytesIO
import base64

class Gen_Plot():
    def __init__(self) -> None:
        pass

    def gen_auc_plot(self, res_dic):
        fig = plt.figure(figsize=(6, 12), dpi=100)
        fig.subplots_adjust(hspace=0.5) 

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(res_dic['fpr'], res_dic['tpr'], label=f"AUC = {res_dic['auc_roc']:.4f}", color='royalblue')
        ax1.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
        ax1.set_ylabel('True Positive Rate', fontsize=14, labelpad=15)
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        ax1.legend(loc='best')

        cf_matrix = confusion_matrix(res_dic['y_true'], res_dic['y_pred'])
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax2)
        ax2.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax2.set_xlabel('Predicted', fontsize=14, labelpad=15)
        ax2.xaxis.set_ticklabels(['Positive', 'Negative'])
        ax2.set_ylabel('Actual', fontsize=14, labelpad=15)
        ax2.yaxis.set_ticklabels(['Positive', 'Negative'])

        buf = BytesIO()  
        fig.savefig(buf, format="png")
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot
    
    def uni_data_visualise(self, dataset):
        df = pd.read_csv(dataset)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        anomalies_df = X[y == 1]
        non_anomalies_df = X[y == 0]

        fig = plt.figure(figsize=(8, 12), tight_layout=True, dpi=100)
        fig.subplots_adjust(hspace=0.5) 
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        sns.scatterplot(data=non_anomalies_df['Value'], label='Normal Data', ax=ax1, color='skyblue')
        sns.scatterplot(data=anomalies_df['Value'], label='Anomalies', ax=ax1, color='crimson')
        ax1.set_title('Data with Labeled Anomalies', fontsize=16, pad=20)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')

        sns.histplot(data=df['Value'], kde=True, ax=ax2, color='cornflowerblue')
        ax2.set_title('Data distribution', fontsize=16, pad=20)
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Count')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot