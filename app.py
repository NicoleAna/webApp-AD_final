# Flask app
import pandas as pd

from flask import Flask, render_template, request, redirect, session
from flask_session import Session
from flask_caching import Cache

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
from plots.visuals import Gen_Plot

import io
import secrets
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# configure flask app
app = Flask(__name__)

# secure key for session management
app.secret_key = secrets.token_hex(16)

# configure flask caching
cache = Cache(app, config={'CACHE_TYPE':'SimpleCache', 'CACHE_DEFAULT_TIMEOUT':300})

# configure flask session
app.config["SESSION_PERMANENT"] = True
# session lifetime of 30 mins
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30) 
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

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
    if not session.get("name"):
        return redirect("/login")
    return render_template("index.html", session_name=session.get("name"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["name"] = request.form.get("name")
        return redirect("/")
    return render_template("login.html") 


@app.route("/about", methods=["GET"])
def aboutus():
    if not session.get("name"):
        return redirect("/login")
    return render_template("aboutus.html")
    

@app.route("/input_form")
def ip_form():
    if not session.get("name"):
        return redirect("/login")
    return render_template("input_form.html", algos=ALGO)


@app.route("/visualize", methods=["GET", "POST"])
def visual():
    if not session.get("name"):
        return redirect("/login")
    return render_template("visualize.html", algos=ALGO)


@app.route("/visualizedata", methods=["GET", "POST"])
def datavis():
    if not session.get("name"):
        return redirect("/login")
    return render_template("visualize_data.html")


@app.route("/inputs", methods=["POST"])
def inputs():
    if not session.get("name"):
        return redirect("/login")

    file = request.files.get("dataset")
    algo = request.form.getlist("algo")
    
    if not file:
        return render_template("input_form.html", error="Please upload a CSV file", algos=ALGO, selected_algo=algo)
    
    data = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    dataset = pd.read_csv(data)

    plot_model = Gen_Plot()
    plots = dict()
    selected_algo = dict()

    for model in algo:
        if model == "Generative Adversarial Networks(GAN)":
            gan_model = GAN(dataset)
            gan_model.train(epochs=50, batch_size=32)
            gan_res = gan_model.test()
            selected_algo[model] = gan_res
            plots[model] = plot_model.gen_auc_plot(gan_res)
        
        elif model == "Local Outlier Factor(LOF)":
            lof_model = Lof(dataset)
            lof_res = lof_model.train_test()
            selected_algo[model] = lof_res
            plots[model] = plot_model.gen_auc_plot(lof_res)

        elif model == "Isolation Forest(IForest)":
            iforest_model = iForest(dataset)
            iforest_res = iforest_model.train_test()
            selected_algo[model] = iforest_res
            plots[model] = plot_model.gen_auc_plot(iforest_res)

        elif model == "AutoEncoders":
            auto_model = AutoEncoder(dataset)
            auto_model.auto()
            auto_res = auto_model.train_test(epochs=50, batch_size=64)
            selected_algo[model] = auto_res
            plots[model] = plot_model.gen_auc_plot(auto_res)

        elif model == "DevNet":
            devnet_model = Devnet(dataset)
            devnet_res = devnet_model.train_test(epochs=10)
            selected_algo[model] = devnet_res
            plots[model] = plot_model.gen_auc_plot(devnet_res)

        elif model == "Elliptic Envelope":
            env_model = ellipticEnvelope(dataset)
            env_res = env_model.train_test()
            selected_algo[model] = env_res
            plots[model] = plot_model.gen_auc_plot(env_res)

        elif model == "DAGMM":
            dagmm_model = Dagmm1(dataset)
            dagmm_res = dagmm_model.train_test()
            selected_algo[model] = dagmm_res
            plots[model] = plot_model.gen_auc_plot(dagmm_res)

        elif model == "Quantile Regression":
            qreg_model = QReg(dataset)
            qreg_model.build_model()
            qreg_res = qreg_model.train_test()
            selected_algo[model] = qreg_res
            plots[model] = plot_model.gen_auc_plot(qreg_res)

        elif model == "Long Short Term Memory(LSTM)":
            lstm_model = Lstm(dataset)
            lstm_model.build_model()
            lstm_res = lstm_model.train_test(epochs=50, batch_size=64)
            selected_algo[model] = lstm_res
            plots[model] = plot_model.gen_auc_plot(lstm_res)

        elif model == "MGBTAI":
            mgbtai_model = MGBTAI(dataset)
            mgbtai_model.train_mgbtai()
            mgbtai_res = mgbtai_model.evaluate_mgbtai()
            selected_algo[model] = mgbtai_res
            plots[model] = plot_model.gen_auc_plot(mgbtai_res)

        elif model == "DBTAI":
            dbtai_model = DBTAI(dataset)
            dbtai_model.train_dbtai()
            dbtai_res = dbtai_model.evaluate_dbtai()
            selected_algo[model] = dbtai_res
            plots[model] = plot_model.gen_auc_plot(dbtai_res)
        
        else:
            return render_template("visualize.html", error="Some error occured", algos=ALGO)
        
    if len(plots) != 0:
        if len(plots) == 1:    
            return render_template("visualize.html", algos=algo, plot=plots)
        else:
            auc_plots = plot_model.comp_auc(selected_algo)
            return render_template("visualize.html", algos=algo, plot=plots, auc_plot=auc_plots)
    else:
        return render_template("input_form.html", error="Some error occured", algos=ALGO)


@app.route("/datavis", methods=["POST"])
def dataVis():
    if not session.get("name"):
        return redirect("/login")
    file = request.files.get("dataset")
    if not file:
        return render_template("visualize_data.html", error="Please upload a csv file")
    
    dataset = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    plot_graph = Gen_Plot()
    plot = plot_graph.uni_data_visualise(dataset)

    if plot:    
        return render_template("visualize_data.html", plot=plot, submitted=True)
    else:
        return render_template("visualize_data.html", error="Some error occured")


@app.route("/logout")
def logout():
    cache.clear()
    session["name"] = None
    return redirect("/")