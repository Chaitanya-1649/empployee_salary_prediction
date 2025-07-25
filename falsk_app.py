from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

education_labels = ['HS-grad', 'Bachelors', 'Masters']
occupation_labels = ['Tech-support', 'Sales', 'Exec-managerial']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_data = None

    if request.method == "POST":
        age = int(request.form["age"])
        education = int(request.form["education"])
        occupation = int(request.form["occupation"])
        hours = int(request.form["hours"])
        experience = int(request.form["experience"])

        input_features = np.array([[age, education, occupation, hours, experience]])
        output = model.predict(input_features)[0]

        prediction = ">50K" if output == 1 else "<=50K"
        user_data = {
            'age': age,
            'education': education_labels[education],
            'occupation': occupation_labels[occupation],
            'hours': hours,
            'experience': experience
        }

    return render_template("index.html", prediction=prediction, user_data=user_data)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)
