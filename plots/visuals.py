from matplotlib.figure import Figure
import seaborn as sns

from io import BytesIO
import base64

class Gen_Plot():
    def __init__(self, fpr, tpr, auc_roc) -> None:
        self.tpr = tpr
        self.fpr = fpr
        self.auc_roc = auc_roc

    def gen_plot(self):
        fig = Figure()
        ax = fig.subplots()
        ax.plot(self.fpr, self.tpr, label=f'AUC = {self.auc_roc:.4f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='best')
        buf = BytesIO()  
        fig.savefig(buf, format="png")
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot