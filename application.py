from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

@application.route("/")
def index():
    return "Application is running:)"
@application.route("/api")
def load_model(text):
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)
    
    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    
    # model making prediction - output will be 'FAKE' if fake, else 'REAL'
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]
    if(prediction == 'FAKE'):
        return 0
    return 1

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['sentence']
        predictionval = load_model(user_input)
        prediction = None
        if predictionval == 0:
            prediction = "This is fake news"
        else:
           prediction = "This is real news" 

    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    application.run(port=5000, debug=True)