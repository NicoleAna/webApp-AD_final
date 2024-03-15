import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO
import base64

class Gen_Plot1():
    def __init__(self, dataset) -> None:
        self.df = pd.read_csv(dataset)
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        self.anomalies_df = X[y == 1]
        self.non_anomalies_df = X[y == 0]

    def scatterplot(self):
        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True, dpi=100)
        sns.scatterplot(data=self.non_anomalies_df.iloc[:, 0], label='Normal Data', ax=ax, color='skyblue')
        sns.scatterplot(data=self.anomalies_df.iloc[:, 0], label='Anomalies', ax=ax, color='crimson')
        ax.set_title('Scatter Plot', fontsize=20, pad=20)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot

    def histogram(self):
        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True, dpi=100)
        sns.histplot(data=self.df.iloc[:, :-1], kde=True, ax=ax, color='cornflowerblue')
        ax.set_title('Histogram with KDE', fontsize=20, pad=20)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot
    
    def boxplot(self):
        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True, dpi=100)
        sns.boxplot(data=self.df.iloc[:, :-1], orient='h', ax=ax, palette='mako')
        ax.set_title('Boxplot', fontsize=20, pad=20)
        ax.set_xlabel('Value')

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot