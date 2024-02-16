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


ALGO = {
    "Generative Adversarial Networks(GAN)": GAN,
    "Local Outlier Factor(LOF)": Lof,
    "Isolation Forest(IForest)": iForest,
    "AutoEncoders": AutoEncoder,
    "Elliptic Envelope": ellipticEnvelope,
    "DAGMM": Dagmm1,
    "Quantile Regression": QReg,
    "Long Short Term Memory(LSTM)": Lstm,
    "MGBTAI": MGBTAI,
    "DBTAI": DBTAI
}


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
    #x=5
    plot_model = Gen_Plot()
    plots = dict()
    selected_algo = dict()

    for model in algo:
        if model not in ALGO:
            return render_template("visualize.html", error="Invalid model selected", algos=ALGO.keys())
        
        model_class = ALGO[model]
        model_instance = model_class(dataset)
        subplot = []

        if hasattr(model_instance, 'train'):
            model_instance.train()
            res = model_instance.test()
        elif hasattr(model_instance, 'build_model'):
            model_instance.build_model()
            res = model_instance.train_test()
        else:
            res = model_instance.train_test()

        selected_algo[model] = res
        subplot.append(plot_model.gen_auc_plot(res))
        subplot.append(plot_model.gen_confusion_matrix(res))
        plots[model] = subplot
    
    if plots:
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