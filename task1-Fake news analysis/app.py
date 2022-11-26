from flask import Flask,render_template, request, Markup
import pandas as pd
import warnings
import pickle
import re
import string
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = 'your secret key'
warnings.filterwarnings("ignore")
model = pickle.load(open(r'C:\projects\task1-Fake news analysis\Model\Data\best_model.pickle', 'rb'))
vector = pickle.load(open(r'C:\projects\task1-Fake news analysis\Model\Data\vectorizing.pickle', 'rb'))


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text) #removing URLs
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


@app.route("/", methods=['GET', 'POST'])
def predire():
    msg=''
    if request.method == 'POST':
        text = request.form['txtMsg']
        news = {"text":[text]}
        new_df = pd.DataFrame(news)
        new_df['text']=new_df['text'].apply(wordopt)
        new_df=new_df['text']
        vectorizing=vector.transform(new_df)
        prediction = model.predict(vectorizing)
        if prediction == 0:
            msg= Markup("""<div class="alert alert-danger" role="alert">THIS IS A FAKE NEWS</div>""")
        else:
            msg = Markup("""<div class="alert alert-success" role="alert">THIS  A REAL NEWS</div>""")
    return render_template('WelcomePage.html', msg=msg)


if __name__ == '__main__':
    app.run()