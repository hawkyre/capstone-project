from nltk.tokenize import word_tokenize
import re
import numpy as np
import gensim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

eng_stopwords = stopwords.words('english')

topics = ['depression',
          'anxiety',
          'parenting',
          'self-esteem',
          'relationship-dissolution',
          'spirituality',
          'trauma',
          'domestic-violence',
          'anger-management',
          'intimacy',
          'addiction',
          'family-conflict',
          'marriage',
          'relationships',
          'behavioral-change',
          'stress',
          'self-harm',
          'counseling-fundamentals']


class TextToContext:
    def __init__(self):
        self.model = gensim.similarities.Similarity.load(
            'ttc_models/text_to_context.model')

        self.dictionary = gensim.corpora.Dictionary.load(
            'ttc_models/text_to_context.dictionary')

        self.tf_idf = gensim.models.TfidfModel.load(
            'ttc_models/text_to_context.tfidf')

        self.lmtz = WordNetLemmatizer()
        self.stmr = PorterStemmer()

    def predict(self, text):
        query_doc = self.preprocess_sentence(text)
        query_doc_bow = self.dictionary.doc2bow(query_doc)
        # perform a similarity query against the corpus
        query_doc_tf_idf = self.tf_idf[query_doc_bow]
        # print(document_number, document_similarity)
        pred = self.model[query_doc_tf_idf]

        topic_prob = list(zip(topics, pred))
        sorted_context = sorted(topic_prob, key=lambda x: x[1], reverse=True)
        print(sorted_context)

    def preprocess_sentence(self, sentence):
        sentence = [w.lower() for w in word_tokenize(sentence)]
        sentence = [word for word in sentence if word not in eng_stopwords]
        sentence = [self.stmr.stem(self.lmtz.lemmatize(word))
                    for word in sentence]
        sentence = [word for word in sentence if re.match(r"^[a-z]+$", word)]
        return sentence
