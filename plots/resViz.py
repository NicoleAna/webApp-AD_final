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
        ax.plot(res_dic['fpr'], res_dic['tpr'], label=f"Precision = {res_dic['precision']:.4f}", color='royalblue')
        ax.plot(res_dic['fpr'], res_dic['tpr'], label=f"Recall = {res_dic['recall']:.4f}", color='royalblue')
        ax.plot(res_dic['fpr'], res_dic['tpr'], label=f"F1 Score = {res_dic['f1_score']:.4f}", color='royalblue')
        ax.plot(res_dic['fpr'], res_dic['tpr'], label=f"AUC ROC = {res_dic['auc_roc']:.4f}", color='royalblue')


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
            # ax.plot(metric['fpr'], metric['tpr'], label=f"{model} Precision = {metric['precision']:.4f}")
            # ax.plot(metric['fpr'], metric['tpr'], label=f"{model} Recall = {metric['recall']:.4f}")
            # ax.plot(metric['fpr'], metric['tpr'], label=f"{model} F1 Score = {metric['auc_roc']:.4f}")
            ax.plot(metric['fpr'], metric['tpr'], label=f"{model} AUC roc = {metric['auc_roc']:.4f}")
            ax.plot(metric['fpr'], metric['tpr'], label=f"{model} AUC roc = {metric['auc_roc']:.4f}")


        ax.set_xlabel('False Positive Rate', fontsize=14, labelpad=15)
        ax.set_ylabel('True Positive Rate', fontsize=14, labelpad=15)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))

        buf = BytesIO()  
        fig.savefig(buf, format="png", bbox_inches='tight')
        plot = base64.b64encode(buf.getbuffer()).decode("ascii") 
        return plot
    
    def all_auc_roc(self,all_auc_roc):

        plt.figure(figsize=(8, 6))
        plt.title('Receiver Operating Characteristic', fontsize=20)

        # Plot ROC curves and AUC values for each algorithm
        plt.plot(fpr3, tpr3, 'orange', linewidth=2, label='LOF AUC = %0.2f' % auc_roc_lof)
        plt.plot(fpr1, tpr1, 'yellow', linewidth=2, label='IForest AUC = %0.2f' % auc_roc_if)
        plt.plot(fpr8, tpr8, 'lime', linewidth=2, label='AutoEncoders AUC = %0.2f' % auc_roc_auto)
        plt.plot(fpr5, tpr5, 'cyan', linewidth=2, label='DAGMM AUC = %0.2f' % auc_roc_dagmm)
        plt.plot(fpr2, tpr2, 'g', linewidth=2, label='Envelope AUC = %0.2f' % auc_roc_env)
        plt.plot(fpr7, tpr7, 'blue', linewidth=2, label='DevNet AUC = %0.2f' % auc_roc_devnet)
        plt.plot(fpr6, tpr6, 'magenta', linewidth=2, label='GAN AUC= %0.2f' % auc_roc_gan)
        plt.plot(fpr9, tpr9, color="red", linewidth=2, label='MGBTAI AUC = %0.2f' % auc_roc_mg)
        plt.plot(fpr10, tpr10, 'black', linewidth=2, label='DBTAI AUC = %0.2f' % auc_roc_db)

        # Customize the plot
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=18)

        # Move the legend below the plot
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), prop={'size': 14}, ncol=2) # Adjust ncol as needed
        plt.legend(loc='upper center', bbox_to_anchor=(0.88, 0.3), prop={'size': 7}, ncol=1)

        # Save the plot as a JPG image
        plt.grid()
        plt.tight_layout()
        plt.savefig("roc_plot.jpg", dpi=300, bbox_inches='tight')  # Specify the filename and DPI
        plt.show()


    def gen_bar_graph(self,precision):
        # Define algorithms and their corresponding AUC values
        algorithms = ['LOF', 'IForest', 'AutoEncoders', 'DAGMM', 'Envelope', 'DevNet', 'GAN', 'MGBTAI', 'DBTAI']
        auc_values = [auc_roc_lof, auc_roc_if, auc_roc_auto, auc_roc_dagmm, auc_roc_env, auc_roc_devnet, auc_roc_gan, auc_roc_mg, auc_roc_db]

        # Create a bar graph
        plt.figure(figsize=(10, 6))
        plt.barh(algorithms, auc_values, color='skyblue')
        plt.xlabel('AUC Value', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)
        plt.title('AUC Values of Different Algorithms', fontsize=16)

        # Display AUC values on bars
        for index, value in enumerate(auc_values):
            plt.text(value, index, '%.2f' % value, ha='left', fontsize=10)

        plt.gca().invert_yaxis()  # Invert y-axis to display algorithms from top to bottom
        plt.tight_layout()
        plt.show()

        # Create a bar graph
        plt.figure(figsize=(10, 6))
        plt.barh(algorithms, auc_values, color='skyblue')
        plt.xlabel('AUC Value', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)
        plt.title('AUC Values of Different Algorithms', fontsize=16)

        # Display AUC values on bars
        for index, value in enumerate(auc_values):
            plt.text(value, index, '%.2f' % value, ha='left', fontsize=10)

        plt.gca().invert_yaxis()  # Invert y-axis to display algorithms from top to bottom
        plt.tight_layout()
        plt.show()

        # Create a bar graph
        plt.figure(figsize=(10, 6))
        plt.barh(algorithms, auc_values, color='skyblue')
        plt.xlabel('AUC Value', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)
        plt.title('AUC Values of Different Algorithms', fontsize=16)

        # Display AUC values on bars
        for index, value in enumerate(auc_values):
            plt.text(value, index, '%.2f' % value, ha='left', fontsize=10)

        plt.gca().invert_yaxis()  # Invert y-axis to display algorithms from top to bottom
        plt.tight_layout()
        plt.show()

        # Create a bar graph
        plt.figure(figsize=(10, 6))
        plt.barh(algorithms, auc_values, color='skyblue')
        plt.xlabel('AUC Value', fontsize=14)
        plt.ylabel('Algorithm', fontsize=14)
        plt.title('AUC Values of Different Algorithms', fontsize=16)

        # Display AUC values on bars
        for index, value in enumerate(auc_values):
            plt.text(value, index, '%.2f' % value, ha='left', fontsize=10)

        plt.gca().invert_yaxis()  # Invert y-axis to display algorithms from top to bottom
        plt.tight_layout()
        plt.show()