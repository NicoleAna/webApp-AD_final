# Flask app
import pandas as pd

from flask import Flask, render_template, request

from models.gan import GAN
from models.lof import Lof
from models.iforest import iForest
from models.autoencoders import AutoEncoder
from models.devnet import Devnet
from models.elliptic import ellipticEnvelope
from models.dagmm1 import Dagmm1
from models.quantilereg import QReg
from models.lstm import Lstm
from models.dbtai import DBTAI
from models.mgbtai import MGBTAI
from plots.dataViz import Gen_Plot1
from plots.resViz import Gen_Plot

import io
import warnings

warnings.filterwarnings('ignore')

# configure flask app
app = Flask(__name__)


ALGO = [
    "Generative Adversarial Networks(GAN)", 
    "Local Outlier Factor(LOF)", 
    "Isolation Forest(IForest)", 
    "AutoEncoders", 
    # "DevNet",  
    "Elliptic Envelope", 
    "DAGMM",
    "Quantile Regression", 
    "Long Short Term Memory(LSTM)",
    "MGBTAI",
    "DBTAI"
]


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/about", methods=["GET"])
def aboutus():
    return render_template("aboutus.html")
    

@app.route("/input_form")
def ip_form():
    return render_template("input_form.html", algos=ALGO, loading=False)


@app.route("/visualize", methods=["GET", "POST"])
def visual():
    return render_template("visualize.html", algos=ALGO)


@app.route("/visualizedata", methods=["GET", "POST"])
def datavis():
    return render_template("visualize_data.html")


@app.route("/inputs", methods=["POST"])
def inputs():
    file = request.files.get("dataset")
    algo = request.form.getlist("algo")
    
    if not file:
        return render_template("input_form.html", error="Please upload a CSV file", algos=ALGO, selected_algo=algo)
    
    data = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    dataset = pd.read_csv(data)

    plot_model = Gen_Plot()
    plots = dict()
    selected_algo = dict()

    loading_temp = render_template("input_form.html", loading=True)

    for model in algo:
        if model == "Generative Adversarial Networks(GAN)":
            subplot = list()
            gan_model = GAN(dataset)
            gan_model.train(epochs=50, batch_size=32)
            gan_res = gan_model.test()
            selected_algo[model] = gan_res
            subplot.append(plot_model.gen_auc_plot(gan_res))
            subplot.append(plot_model.gen_confusion_matrix(gan_res))
            plots[model] = subplot
        
        elif model == "Local Outlier Factor(LOF)":
            subplot = list()
            lof_model = Lof(dataset)
            lof_res = lof_model.train_test()
            selected_algo[model] = lof_res
            subplot.append(plot_model.gen_auc_plot(lof_res))
            subplot.append(plot_model.gen_confusion_matrix(lof_res))
            plots[model] = subplot

        elif model == "Isolation Forest(IForest)":
            subplot = list()
            iforest_model = iForest(dataset)
            iforest_res = iforest_model.train_test()
            selected_algo[model] = iforest_res
            subplot.append(plot_model.gen_auc_plot(iforest_res))
            subplot.append(plot_model.gen_confusion_matrix(iforest_res))
            plots[model] = subplot

        elif model == "AutoEncoders":
            subplot = list()
            auto_model = AutoEncoder(dataset)
            auto_model.auto()
            auto_res = auto_model.train_test(epochs=50, batch_size=64)
            selected_algo[model] = auto_res
            subplot.append(plot_model.gen_auc_plot(auto_res))
            subplot.append(plot_model.gen_confusion_matrix(auto_res))
            plots[model] = subplot

        elif model == "DevNet":
            subplot = list()
            devnet_model = Devnet(dataset)
            devnet_res = devnet_model.train_test(epochs=10)
            selected_algo[model] = devnet_res
            subplot.append(plot_model.gen_auc_plot(devnet_res))
            subplot.append(plot_model.gen_confusion_matrix(devnet_res))
            plots[model] = subplot

        elif model == "Elliptic Envelope":
            subplot = list()
            env_model = ellipticEnvelope(dataset)
            env_res = env_model.train_test()
            selected_algo[model] = env_res
            subplot.append(plot_model.gen_auc_plot(env_res))
            subplot.append(plot_model.gen_confusion_matrix(env_res))
            plots[model] = subplot

        elif model == "DAGMM":
            subplot = list()
            dagmm_model = Dagmm1(dataset)
            dagmm_res = dagmm_model.train_test()
            selected_algo[model] = dagmm_res
            subplot.append(plot_model.gen_auc_plot(dagmm_res))
            subplot.append(plot_model.gen_confusion_matrix(dagmm_res))
            plots[model] = subplot

        elif model == "Quantile Regression":
            subplot = list()
            qreg_model = QReg(dataset)
            qreg_model.build_model()
            qreg_res = qreg_model.train_test()
            selected_algo[model] = qreg_res
            subplot.append(plot_model.gen_auc_plot(qreg_res))
            subplot.append(plot_model.gen_confusion_matrix(qreg_res))
            plots[model] = subplot

        elif model == "Long Short Term Memory(LSTM)":
            subplot = list()
            lstm_model = Lstm(dataset)
            lstm_model.build_model()
            lstm_res = lstm_model.train_test(epochs=50, batch_size=64)
            selected_algo[model] = lstm_res
            subplot.append(plot_model.gen_auc_plot(lstm_res))
            subplot.append(plot_model.gen_confusion_matrix(lstm_res))
            plots[model] = subplot

        elif model == "MGBTAI":
            subplot = list()
            mgbtai_model = MGBTAI(dataset)
            mgbtai_model.train_mgbtai()
            mgbtai_res = mgbtai_model.evaluate_mgbtai()
            selected_algo[model] = mgbtai_res
            subplot.append(plot_model.gen_auc_plot(mgbtai_res))
            subplot.append(plot_model.gen_confusion_matrix(mgbtai_res))
            plots[model] = subplot

        elif model == "DBTAI":
            subplot = list()
            dbtai_model = DBTAI(dataset)
            dbtai_model.train_dbtai()
            dbtai_res = dbtai_model.evaluate_dbtai()
            selected_algo[model] = dbtai_res
            subplot.append(plot_model.gen_auc_plot(dbtai_res))
            subplot.append(plot_model.gen_confusion_matrix(dbtai_res))
            plots[model] = subplot
        
        else:
            return render_template("visualize.html", error="Some error occured", algos=ALGO)
        
    
    if len(plots) != 0:
        if len(plots) == 1:    
            return render_template("visualize.html", algos=algo, plot=plots, selected_algo=selected_algo)
        else:
            auc_plots = plot_model.comp_auc(selected_algo)
            return render_template("visualize.html", algos=algo, plot=plots, auc_plot=auc_plots, selected_algo=selected_algo)
    else:
        return render_template("input_form.html", error="Some error occured", algos=ALGO)


@app.route("/datavis", methods=["POST"])
def dataVis():
    file = request.files.get("dataset")
    if not file:
        return render_template("visualize_data.html", error="Please upload a csv file")
    
    plot = list()
    dataset = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    plot_graph = Gen_Plot1(dataset)
    plot.append(plot_graph.scatterplot())
    plot.append(plot_graph.histogram())

    if plot:    
        return render_template("visualize_data.html", plot=plot, submitted=True)
    else:
        return render_template("visualize_data.html", error="Some error occured")