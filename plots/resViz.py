
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
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.plot(res_dic['fpr'], res_dic['tpr'], label=f"AUC = {res_dic['auc_roc']:.4f}", color='royalblue')
        ax.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=14, labelpad=15)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot

    def gen_confusion_matrix(self, res_dic):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        cf_matrix = confusion_matrix(res_dic['y_true'], res_dic['y_pred'])
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Predicted', fontsize=14, labelpad=15)
        ax.xaxis.set_ticklabels(['Positive', 'Negative'])
        ax.set_ylabel('Actual', fontsize=14, labelpad=15)
        ax.yaxis.set_ticklabels(['Positive', 'Negative'])

        
        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot
    
    def comp_auc(self, selected_algo):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

        for model, metric in selected_algo.items():
            ax.plot(metric['fpr'], metric['tpr'], label=f"{model} AUC = {metric['auc_roc']:.4f}")

        ax.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=14, labelpad=15)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot
    
    def comp_precision(self, selected_algo):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

        models = []
        precisions = []

        for model, metric in selected_algo.items():
            precision = metric['precision']
            models.append(model)
            precisions.append(precision)

        ax.bar(models, precisions, color='skyblue')

        ax.set_xlabel('Algorithms', fontsize=14, labelpad=15)
        ax.set_ylabel('Precision', fontsize=14, labelpad=15)
        ax.set_title('Precision of Different Algorithms', fontsize=16, pad=20)
        ax.tick_params(axis='x', rotation=45)

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        plt.close(fig)  # Close the figure to release memory
        return plot
    
    def comp_recall(self, selected_algo):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

        models = []
        recalls = []

        for model, metric in selected_algo.items():
            recall = metric['recall']
            models.append(model)
            recalls.append(recall)

        ax.bar(models, recalls, color='crimson')

        ax.set_xlabel('Algorithms', fontsize=14, labelpad=15)
        ax.set_ylabel('Recall', fontsize=14, labelpad=15)
        ax.set_title('Recall of Different Algorithms', fontsize=16, pad=20)
        ax.tick_params(axis='x', rotation=45)

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        plt.close(fig)  # Close the figure to release memory
        return plot
    
    def comp_f1(self, selected_algo):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

        models = []
        f1s = []

        for model, metric in selected_algo.items():
            f1 = metric['f1_score']
            models.append(model)
            f1s.append(f1)

        ax.bar(models, f1s, color='mediumseagreen')

        ax.set_xlabel('Algorithms', fontsize=14, labelpad=15)
        ax.set_ylabel('Precision', fontsize=14, labelpad=15)
        ax.set_title('F1-Score of Different Algorithms', fontsize=16, pad=20)
        ax.tick_params(axis='x', rotation=45)

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        plt.close(fig)  # Close the figure to release memory
        return plot