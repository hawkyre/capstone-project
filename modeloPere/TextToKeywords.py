# import dependencies
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

stop_words = stopwords.words('english')

class TextToKeywords:
    def __init__(self):
        self.model = models.ldamodel.LdaModel.load(
            'ttk_models/text_to_keywords.model')
        
        self.dictionary = gensim.corpora.Dictionary.load(
            'ttk_models/text_to_keywords.dictionary')

        self.topics = self.model.show_topics(num_topics= -1, num_words=10, formatted=False)
        self.topics = sorted(self.topics, key=lambda x: x[0])
        self.topics = list(map(lambda x: [y[0] for y in x[1]], self.topics))
        
        self.stemmer = PorterStemmer()
        self.lmtz = WordNetLemmatizer()

    def preprocess_sentence(self, text):
        """
        Function to clean text of websites, email addresess and any punctuation
        We also lower case the text
        """
        text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
        #text = re.sub(r'\-', ' ', text)
        text = re.sub("[^a-zA-Z ]", "", text)
        text = text.lower() # lower case the text
        text = nltk.word_tokenize(text)
        
        text = [word for word in text if word not in stop_words]

    
        try:
            text = [self.lmtz.lemmatize(word) for word in text]
            text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
            text = self.dictionary.doc2bow(text)
        except IndexError: # the word "oed" broke this, so needed try except
            pass

        return text
    
    def keep_to_k_words(self, text, k=1400):
        fdist = FreqDist(text)
        top_k_words,_ = zip(*fdist.most_common(k))
        top_k_words = set(top_k_words)
        return [word for word in text if word in top_k_words]
    
    def predict(self, text):

        text = self.keep_to_k_words(self.preprocess_sentence(text))
        pred = self.model[text]
        pred = list(map(lambda x: x[1], pred))
        topic_pred = list(zip(self.topics, pred))
        sorted_keywords = sorted(topic_pred, key=lambda x: x[1], reverse=True)
        print(sorted_keywords[0][0])
        