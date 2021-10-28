from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import string
from nltk.corpus import stopwords


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/sentiment_tweets3.csv")
    df_data = df[["message to examine", "label (depression result)"]]
    #Features and labels
    df_text= df_data['message to examine']
    df_labels= df_data['label (depression result)']
    # Define a function that removes punctuations and stopwords
    def text_process(tex):
        
        # Check characters to see if they are in punctuation
        no_punc = [char for char in tex if char not in string.punctuation]
        
        # Join the characters again to form the string.
        no_punc = ''.join(no_punc)

        # Remove stopwords and return cleaned text
        return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

    Log_pipeline = Pipeline([('Log_bow', CountVectorizer(analyzer=text_process)),
                        ('Log_tfidf', TfidfTransformer()),
                        ('Log_classifier', LogisticRegression())
                       ])
    #Features
    doc = df_text
    vert = Log_pipeline[:2]
    trans_X = vert.fit_transform(doc)

    text_train, text_test, label_train, label_test = train_test_split(trans_X,df_labels, test_size=0.3)

    #Train and test
    Log_pipeline[2].fit(text_train, label_train)
    Log_pipeline[2].score(text_test, label_test)


    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = vert.transform(data).toarray()
        predict = Log_pipeline[2].predict(vect)
    return render_template('result.html', prediction = predict)





if __name__ == '__main__':
    app.run(debug=True)