import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns

from io import BytesIO
import base64

class Gen_Plot():
    def __init__(self) -> None:
        pass

    def gen_auc_plot(self, fpr, tpr, auc_roc):
        fig = Figure()
        ax = fig.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='best')
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

        fig = Figure()
        ax = fig.subplots()
        sns.scatterplot(data=non_anomalies_df['Value'], label='Normal Data', ax=ax)
        sns.scatterplot(data=anomalies_df['Value'], label='Anomalies', ax=ax)
        ax.set_title('Data with Labeled Anomalies')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot