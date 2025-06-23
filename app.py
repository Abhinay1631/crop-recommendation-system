from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Multilingual crop dictionary
crop_dict = {
    "english": {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    },
    "hindi": {
        1: "चावल", 2: "मक्का", 3: "जूट", 4: "कपास", 5: "नारियल", 6: "पपीता", 7: "संतरा",
        8: "सेब", 9: "खरबूजा", 10: "तरबूज", 11: "अंगूर", 12: "आम", 13: "केला",
        14: "अनार", 15: "मसूर", 16: "उड़द", 17: "मूंग", 18: "माथ बीन्स",
        19: "अरहर", 20: "राजमा", 21: "चना", 22: "कॉफी"
    },
    "telugu": {
        1: "బియ్యం", 2: "మొక్కజొన్న", 3: "జూట్", 4: "పత్తి", 5: "కొబ్బరి", 6: "బొప్పాయి", 7: "నారింజ",
        8: "ఆపిల్", 9: "కర్బూజ", 10: "దొడ్డపండ్లు", 11: "ద్రాక్ష", 12: "మామిడి", 13: "అరటి",
        14: "దానిమ్మ", 15: "మసూర్", 16: "మినుములు", 17: "పెసర్లు", 18: "మోత్ బీన్స్",
        19: "కందులు", 20: "రాజ్మా", 21: "సెనగలు", 22: "కాఫీ"
    }
}

@app.route('/')
def language_select():
    return render_template("language.html")

@app.route('/set_language', methods=['POST'])
def set_language():
    lang = request.form['language']
    session['lang'] = lang
    if lang == 'english':
        return redirect('/index')
    elif lang == 'hindi':
        return redirect('/index_hindi')
    elif lang == 'telugu':
        return redirect('/index_telugu')

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/index_hindi')
def index_hindi():
    return render_template("index_hindi.html")

@app.route('/index_telugu')
def index_telugu():
    return render_template("index_telugu.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)[0]

    lang = session.get('lang', 'english')
    crop = crop_dict[lang].get(prediction, "Sorry, we could not determine the best crop.")

    if lang == 'english':
        return render_template('index.html', result=crop)
    elif lang == 'hindi':
        return render_template('index_hindi.html', result=crop)
    elif lang == 'telugu':
        return render_template('index_telugu.html', result=crop)

@app.route('/instructions')
def instructions():
    lang = session.get('lang', 'english')
    if lang == 'english':
        return render_template("instructions.html")
    elif lang == 'hindi':
        return render_template("instructions_hindi.html")
    elif lang == 'telugu':
        return render_template("instructions_telugu.html")
@app.route('/reference')
def reference():
    lang = session.get('lang')
    if not lang:
        return redirect('/')
    if lang == 'english':
        return render_template("reference.html")
    elif lang == 'hindi':
        return render_template("reference_hindi.html")
    elif lang == 'telugu':
        return render_template("reference_telugu.html")


if __name__ == "__main__":
    app.run(debug=True)
