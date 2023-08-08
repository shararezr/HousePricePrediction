import model# Import the python file containing the ML model
from flask import Flask, request, render_template,jsonify # Import flask libraries
import numpy as np
# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/')
def home():
    return render_template('index.html') # Render home.html


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        prediction = model.ValuePredictor(to_predict_list)
        average = model.GetMeanHousePrice()
        return render_template("result.html", prediction = prediction, average = average)




# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)