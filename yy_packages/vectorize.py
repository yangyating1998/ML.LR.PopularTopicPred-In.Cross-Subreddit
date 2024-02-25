# Vectorize
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# input: X_train, X_text
# output: vectorized X's

class text_preparation:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.stop_words = set(stopwords.words('english'))

    def stemming_tokenizer(self, text):
        stemmer = PorterStemmer()
        words = word_tokenize(text.lower())
        stemmed_words = [stemmer.stem(word) for word in words if word not in self.stop_words]
        return stemmed_words


    def lemmatizing_tokenizer(self, text):
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text.lower())
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return lemmatized_words

    def vectorizing(self, model = 'default'):
        if model == 'default':
            vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.8, ngram_range=(1,3))
        elif model == 'stemming':
            vectorizer = CountVectorizer(tokenizer=self.stemming_tokenizer, min_df=5, max_df=0.8, ngram_range=(1,3))
        else:
            vectorizer = CountVectorizer(tokenizer=self.lemmatizing_tokenizer, min_df=5, max_df=0.8, ngram_range=(1,3))
        vectorized_X_train = vectorizer.fit_transform(self.X_train)
        vectorized_X_test = vectorizer.transform(self.X_test)
        text_features = list(vectorizer.get_feature_names_out())
        return vectorized_X_train, vectorized_X_test, text_features