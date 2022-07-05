from operator import index
from random import sample
from flask import Flask, redirect, render_template, request
from numpy import not_equal
from model import *

app = Flask(__name__)
read_data = False
preprocess_data = False

@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET', "POST"])
def home():
    return redirect("/classify")
    '''
    global read_data
    data = {}
    if request.method == "POST":
        print("Loading dataset in home")
        if read_data == False:
            print("home reading")
            read_dataset()
            read_data = True
        n_row, n_col = get_data_shape()
        data = {
            "rows" : n_row,
            "cols" : n_col,
            "tables" :  [get_data_head().to_html(classes='data', header="true"), 
                         get_data_tail().to_html(classes='data', header="true")],
            "label_counts": get_label_counts().to_html(header="true", index = None)
        }
        # print(data)    
    return render_template("home.html", data = data)
    '''

@app.route("/preprocess", methods=['GET', 'POST'])
def preprocess():
    global read_data, preprocess_data
    data = {}

    if request.method == "POST":
        if request.form["preprocess"] == "Start Pre-Processing":
            print("preprocess post")
            if read_data == False:
                print("pre-processing read")
                read_dataset()
                read_data = True
            print(preprocess_data)
            if preprocess_data == False:
                print("removing timestamp")
                drop_timestamp()
                print("datatype casting")
                datatype_casting()
                print("null value removing")
                data_cleaning()
                preprocess_data = True

            data = {
                "tables": [data_describe().to_html(classes="data", header="true")]
            }
    # print(data)
    return render_template("preprocess.html", data = data)

@app.route("/featuresel/<method_name>")
def featuresel(method_name):
    global read_data
    if read_data == False:
        print("data not read")
        # return("data not read")
    data = {}
    data["method_name"] = method_name
    if request.args:
        if request.args["method"] == "chi":
            print("Chi-squard test")
            comp = int(request.args["comp"])
            data = {"tables": [chi(comp).to_html(classes="data", header="true")]}
            data["method_name"] = "chi"
            data["comp"] = comp

        elif request.args["method"] == "pca":
            print("pca")
            n_variance = float(request.args["variance"])
            ex_var_rt = pca(n_variance)
            attr_count = len(ex_var_rt)
            data["result"]      = True
            data["method_name"] = "pca"
            data["variance"]    = n_variance
            data["ex_var_rt"]   = ex_var_rt
            data["length"]      = len(ex_var_rt)
 
        elif request.args["method"] == "rfi":
            print("random Forest Importance")
            n_estimator = int(request.args["n_estimator"])
            sel_feat = rfi(n_estimator)
            data["result"]      = True
            data["method_name"] = "rfi"
            data["n_estimator"] = n_estimator
            data["sel_feat"]    = sel_feat
            data["length"]      = len(sel_feat)

    print(data)
    return render_template("featuresel.html", data = data)


@app.route("/decisiontree/<type>")
def decisiontree(type):
    data = {"type": type}
    if type == "binary":
        result = classification("binary", "chi", "gini")
        print(result)
        data["result"] = result

    print(data)
    return render_template("decisiontree.html", data = data)

@app.route("/classify", methods=['GET', 'POST'])
def classify():
    data = {}
    sample_df = get_sample_df()
    if request.method == "POST":
        c = request.form["classify"]
        a_b, a_m, p_b, p_m = classify_data(c)
        data = {"classify": c,
            "actual_b": a_b,
            "actual_m": a_m,
            "pred_b"  : p_b,
            "pred_m"  : p_m
        }
    data["sample_df"] = sample_df.to_html(classes="table table-success table-sm table-hover", header="true", table_id = "sample_df")
        
    
    print(data)
    return render_template("classify.html", data = data)


if __name__ == "__main__":
    app.run(port=3000, debug=True)