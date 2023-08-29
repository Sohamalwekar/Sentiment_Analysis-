from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the input comment from the form
        comment = request.form.get("Comment")

        # Make the prediction using the model
        result = model.predict([comment])

        # Display the predicted sentiment on the web page
        #return render_template("index.html", result=result[0])
        return render_template('index.html', result=f'The prediction is {result[0]}')

    return render_template("index.html")

if __name__ == "__main__":
    app.run()