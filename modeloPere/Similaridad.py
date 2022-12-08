import platform
import itertools
import gensim.corpora as corpora
import pprint
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
print(platform.platform())
pp = pprint.PrettyPrinter(indent=4)


class Similaridad:
    """
    Esta clase permite calcular la similitud entre una frase del listado de respuestas
    con todos los topics del listado de topics
    """

    def __init__(self):
        df = pd.read_csv("data/model-counsel-chat.csv", encoding='UTF8')
        df.drop(columns="Unnamed: 0", inplace=True)
        # df.head()

        # .drop_duplicates()

        processed_response_list, corpus, dictionary = self.preprocess_text(
            df['answer'])

        topic_words, corpus_t, dictionary_t = self.preprocess_text(
            df['topic'])

        processed_response_list = list(filter(None, processed_response_list))

        topic_words = list(itertools.chain(*topic_words))

        tf_idf, sims = self.create_model(corpus_t, dictionary_t)

        self.model = sims
        self.tf_idf = tf_idf
        self.dictionary = dictionary
        self.data = df

        # self.most_similarity(processed_response_list, self.listado_respuestas,
        #                      self.listado_topics, tf_idf, dictionary, sims)

    def parsear_frases(self, texto_a_preprocesar):
        for sentence in texto_a_preprocesar:
            sentence = re.sub(r'\-', ' ', sentence)
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def eliminar_stop_words(self, texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]

    def create_corpus(self, texts):
        # Create Dictionary
        id2word = corpora.Dictionary(texts)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        return corpus

    def create_corpora(self, datos_para_corpora):
        # print(datos_para_corpora)
        return gensim.corpora.Dictionary(datos_para_corpora)

    def preprocess_text(self, texto_a_preprocesar):

        texto_parseado = self.parsear_frases(texto_a_preprocesar)
        texto_preprocesado = self.eliminar_stop_words(texto_parseado)
        # print(texto_preprocesado)

        corpus = self.create_corpus(texto_preprocesado)

        dictionary = self.create_corpora(texto_preprocesado)

        return texto_preprocesado, corpus, dictionary

    # FUNCIÓN QUE CREA MODELO TFIDF
    def create_model(self, corpus, dictionary):
        tf_idf = gensim.models.TfidfModel(corpus)

        sims = gensim.similarities.Similarity('data/', tf_idf[corpus],
                                              num_features=len(dictionary))
        return tf_idf, sims

        # Función similaridad respuestas-topics
    def most_similarity(self, preprocessed_answer_list, answer_list, topic_list, tf_idf, dictionary, sims):
        contador_respuestas = 0
        # print(preprocessed_answer_list)
        for line in preprocessed_answer_list:  # antes data_words
            # print(line)
            query_doc_bow = dictionary.doc2bow(line)
            # perform a similarity query against the corpus
            query_doc_tf_idf = tf_idf[query_doc_bow]

            print(" ")
            print("-"*30 + "ANSWER" + "-"*30)
            print(answer_list[contador_respuestas])
            print(" ")

            contador_respuestas = contador_respuestas+1
            topics_list = []

            for topic in topic_list:

                corrrelations_list = []
                topics_list.append(topic)
                corrrelations_list.append(sims[query_doc_tf_idf])

            sorted_topics = [x for _, x in sorted(
                zip(corrrelations_list[0], topics_list), reverse=True)]
            sorted_correlations = np.sort(corrrelations_list[0])[::-1]

            if sorted_correlations[0] > 0:
                print("----")
                print("Topics sorted by similarity with the text:")
                print(" ")
                for i in range(len(sorted_topics)):
                    if sorted_correlations[i] > 0:
                        print(
                            f'{sorted_topics[i]}: {round(sorted_correlations[i]*100,2)}%')
                    if i == 2:
                        break
            else:
                print("The answer has not been correlated with any topic.")
            print("")


def main():

    similaridad = Similaridad()
    similaridad.run()


main()
