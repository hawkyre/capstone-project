from transformers import (pipeline, AutoTokenizer,
                          BertForSequenceClassification)
import re
import matplotlib.pyplot as plt
import numpy as np
import copy

ekman_to_feeling = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}

feeling_to_ekman = {'anger': 'anger',
                    'annoyance': 'anger',
                    'disapproval': 'anger',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'nervousness': 'fear',
                    'joy': 'joy',
                    'amusement': 'joy',
                    'approval': 'joy',
                    'excitement': 'joy',
                    'gratitude': 'joy',
                    'love': 'joy',
                    'optimism': 'joy',
                    'relief': 'joy',
                    'pride': 'joy',
                    'admiration': 'joy',
                    'desire': 'joy',
                    'caring': 'joy',
                    'sadness': 'sadness',
                    'disappointment': 'sadness',
                    'embarrassment': 'sadness',
                    'grief': 'sadness',
                    'remorse': 'sadness',
                    'surprise': 'surprise',
                    'realization': 'surprise',
                    'confusion': 'surprise',
                    'curiosity': 'surprise'}

ekman_map_base = {
    "anger": 0,
    "disgust": 0,
    "fear": 0,
    "joy": 0,
    "sadness": 0,
    "surprise": 0
}


class TextToEmotion:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "bhadresh-savani/bert-base-go-emotion")

        model = BertForSequenceClassification.from_pretrained(
            "bhadresh-savani/bert-base-go-emotion")

        self.pipe = pipeline(task="text-classification", model=model,
                             tokenizer=tokenizer, top_k=None)

    @staticmethod
    def ekman_from_emotions(emotions):
        m = []
        for sent in emotions:
            ekman_map = copy.deepcopy(ekman_map_base)
            for emotion in sent:
                if emotion['label'] != 'neutral':
                    ekamn_corresp = feeling_to_ekman[emotion['label']]
                    ekman_map[ekamn_corresp] += emotion['score']

            m.append(ekman_map)

        return m

    def emotion_from_text(self, text, wpm=130):
        sentences = []

        if isinstance(text, (list, tuple, np.ndarray)):
            sentences = text
        else:
            sentences = re.split(re.compile(
                '((?<=[.?!]")|((?<=[.?!])(?!")))\s*'), text)

        wps = wpm/60

    #     print(len([item for sublist in sentences for item in sublist]) + len(sentences)-1)
    #     print(len(text))

        sentences = [x.strip() for x in sentences if x.strip() != ""]

        emotion = self.pipe(sentences)

        sent_length = [len(s.split()) / wps for s in sentences]

        x_offset = [sum(sent_length[:i+1])
                    for i in range(len(sent_length))]

        ekman = TextToEmotion.ekman_from_emotions(emotion)

        return list(zip(emotion, ekman, x_offset))

    def emotion_from_sentence_group(self, sentence_data):
        # Sentences consists of an array of (sentence, start_sec, end_sec)

        sentences = [sentence.strip() for (sentence, _start, _end)
                     in sentence_data if sentence.strip() != ""]

        emotion = self.pipe(sentences)
        x_offset = [(end+start)/2.0 for (_sentence, start, end)
                    in sentence_data]

        ekman = TextToEmotion.ekman_from_emotions(emotion)

        return list(zip(emotion, ekman, x_offset))

    # def emotion_from_text_report(self, text, wpm=130):
    #     sentences = re.split(re.compile("\.|\n"), text)
    #     wps = wpm/60

    # #     print(len([item for sublist in sentences for item in sublist]) + len(sentences)-1)
    # #     print(len(text))

    #     sentences = [x.strip() for x in sentences if x.strip() != ""]
    #     emotion = self.pipe(sentences)
    #     sent_length = [len(s.split())/wps for s in sentences]
    #     sent_length_acc = [sum(sent_length[:i+1])
    #                        for i in range(len(sent_length))]

    #     ekman = TextToEmotion.ekman_from_emotions(emotion)

    #     report = list(zip(emotion, ekman, sent_length_acc))
    #     report_array = []
    #     for sent in report:
    #         report_obj = {
    #             'emotion': emotion,
    #             'ekman': ekman,
    #             'sent_length': sent_length,
    #             'sent_length_acc': sent_length_acc
    #         }

    #         report_array.append(report_obj)

    #     return report_array

    @staticmethod
    def get_y_for_label(stats, given_label):
        return [[label['score'] for label in labels if label.get('label') == given_label][0] for labels in (s[0] for s in stats)]

    @staticmethod
    def get_all_ekman(stats):
        return stats[0][1].keys()

    @staticmethod
    def get_all_labels(stats):
        return [l['label'] for l in stats[0][0] if l['label'] != 'neutral']

    @staticmethod
    def get_y_for_ekman(stats, given_ekman):
        return [[ekmans[ekman] for ekman in ekmans.keys() if ekman == given_ekman][0] for ekmans in (s[1] for s in stats)]

    @staticmethod
    def get_y_for_label(stats, given_label):
        return [[label['score'] for label in labels if label.get('label') == given_label][0] for labels in (s[0] for s in stats)]

    def plot_feeling_from_stats(self, stats, y_min=0.05, min_y_max=0.2):
        all_labels = TextToEmotion.get_all_labels(stats)
        all_ekman = TextToEmotion.get_all_ekman(stats)
        x = np.array([s[2] for s in stats])

        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(2, 1, 1)
        ax_ekman = fig.add_subplot(2, 1, 2)

        to_plot_general = []
        to_plot_ekman = []

        for label in all_labels:
            y = np.array(TextToEmotion.get_y_for_label(stats, label))
            if np.quantile(y, 0.95) > y_min or np.max(y) > min_y_max:
                to_plot_general.append((x, y, label))

        for ekman in all_ekman:
            y = np.array(TextToEmotion.get_y_for_ekman(stats, ekman))
            to_plot_ekman.append((x, y, ekman))

        ax.set_prop_cycle('color', [cm(1.*i/len(to_plot_general))
                          for i in range(len(to_plot_general))])
        ax_ekman.set_prop_cycle(
            'color', [cm(1.*i/len(to_plot_ekman)) for i in range(len(to_plot_ekman))])

        for (x, y, label) in to_plot_general:
            ax.plot(x, y, label=label)

        for (x, y, label) in to_plot_ekman:
            ax_ekman.plot(x, y, label=label)

        ax.set_xlabel('Minutes')
        ax_ekman.set_xlabel('Minutes')
        ax.legend(loc=(1.04, 0))
        ax_ekman.legend(loc=(1.04, 0))

    def text_to_graph(self, text):
        stats = self.emotion_from_text(text)
        self.plot_feeling_from_stats(stats)

    def sentence_group_to_graph(self, sent_group):
        stats = self.emotion_from_sentence_group(sent_group)
        self.plot_feeling_from_stats(stats)

    def text_to_emotion(self, text):
        stats = self.emotion_from_text(text)
        print(stats)
