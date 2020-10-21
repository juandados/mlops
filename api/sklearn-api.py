import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import flask

model_path = "models/logit_games_v1"
model = mlflow.sklearn.load_model(model_path)

app = flask.Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    params = flask.request.args

    if "G1" in params.keys():
        new_row = {f"G{i}": params.get(f"G{i}") for i in range(1,11)}
        new_x = pd.DataFrame.from_dict(new_row, orient="index").transpose()
        data["response"] = str(model.predict_proba(new_x)[0][1])
        data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
