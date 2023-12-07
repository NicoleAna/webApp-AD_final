# flask app
from flask import Flask, render_template, request, redirect, session
from flask_session import Session
from flask_caching import Cache

from models.gan import GAN
from models.lof import Lof
from models.iforest import iForest
from plots.visuals import Gen_Plot

import io

# configure flask app
app = Flask(__name__)

# configure flask caching
cache = Cache(app, config={'CACHE_TYPE':'SimpleCache', 'CACHE_DEFAULT_TIMEOUT':300})

# configure flask session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

ALGO = ["Generative Adversarial Networks(GAN)", "Local Outlier Factor(LOF)", "Isolation Forest(IForest)", "AutoEncoders", "DevNet", "DAGMM", "Elliptic Envelope", "Quantile Regression", "Long Short Term Memory(LSTM)"]


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


@app.route("/visualize", methods=["GET", "POST"])
def visual():
    if not session.get("name"):
        return redirect("/login")
    return render_template("visualize.html", algos=ALGO)


@app.route("/inputs", methods=["POST"])
def inputs():
    if not session.get("name"):
        return redirect("/login")
    
    file = request.files.get("dataset")
    algo = request.form.get("algo")
    if algo not in ALGO:
        return render_template("visualize.html", error="Please select a model", algos=ALGO)
    if not file:
        return render_template("visualize.html", error="Please upload a csv file", algos=ALGO)
    
    dataset = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    # generating unique cache key based on dataset and model
    cache_key = f"{file.filename}-{algo}"

    # check if result is cached
    if cache_key in session:
        result = cache.get(cache_key)
        return render_template("visualize.html", algo=algo, plot=result['plot'], algos=ALGO, submitted=True)

    if algo == "Generative Adversarial Networks(GAN)":
        gan_model = GAN(dataset)
        gan_model.train(epochs=50, batch_size=32)
        y_true, y_pred, fpr, tpr, auc_roc = gan_model.test()

    elif algo == "Local Outlier Factor(LOF)":
        lof_model = Lof(dataset)
        y_true, y_pred, fpr, tpr, auc_roc = lof_model.train_test()

    elif algo == "Isolation Forest(IForest)":
        iforest_model = iForest(dataset)
        y_true, y_pred, fpr, tpr, auc_roc = iforest_model.train_test()

    else:
        return render_template("visualize.html", error="Some error occured", algos=ALGO)
        
    plot_model = Gen_Plot(fpr, tpr, auc_roc)
    plot = plot_model.gen_plot()

    # cache the result with timeout of 300s
    cache.set(cache_key, {'auc_roc':auc_roc, 'plot':plot}, timeout=300)
    session[cache_key] = {'auc_roc':auc_roc, 'plot':plot}
    
    if plot:    
        return render_template("visualize.html", algo=algo, auc_roc=auc_roc, plot=plot, algos=ALGO, submitted=True)
    else:
        return render_template("visualize.html", error="Some error occured", algos=ALGO)

  
@app.route("/logout")
def logout():
    cache.clear()
    session["name"] = None
    return redirect("/")