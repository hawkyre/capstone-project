from text_to_emotion import TextToEmotion
import numpy as np
import matplotlib.pyplot as plt
import copy 
import math 
import seaborn as sns
import pandas as pd

ekman_map_base = {
    "anger": [],
    "disgust": [],
    "fear": [],
    "happiness": [],
    "sadness": [],
    "surprise": [],
    "neutral": [],
}

def normalize(num):
    # return math.exp(num/4) / (1+math.exp(num/4))
    return max(0, math.tanh(num/4))

class Plotter:

    @staticmethod
    def image_data_to_points(image_data):
        if len(image_data) == 0:
            return [], []

        sequences = copy.deepcopy(ekman_map_base)
        x = []
        for img in image_data:
            xi = img['x']
            x.append(xi)
            for label in img['y']:
                value = img['y'][label]

                # Normalize from (-inf, +inf) to (0, 1)
                value = normalize(value)
                sequences[label.lower()].append(value)

        return x, sequences

    @staticmethod
    def plot_covariance_matrix(data):
        number_of_samples = 100

        x_group, y_group, labels = data.T

        new_x = []
        new_y = []

        for i in range(len(x_group)):
            x = np.array(x_group[i])
            y = np.array(y_group[i])

            xn = np.linspace(x.min(), x.max(), number_of_samples)
            yn = np.interp(xn, x, y)

            # if 'Face' in labels[i]:
            #     ax.plot(xn, yn, label=labels[i])
            #     ax.scatter(xn, yn, label=labels[i])

            new_x.append(xn)
            new_y.append(yn)
        
        # ax.set_title('Ekman normalized')
        # ax.plot()
        df = pd.DataFrame(np.array(new_y).T, columns=labels)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        image_text_corr_mat = df.corr().to_numpy()[6:12, :6]

        sns.heatmap(image_text_corr_mat, vmin=-1, vmax=1, ax=ax, linewidth=.5, xticklabels=df.columns[:6], yticklabels=df.columns[6:12], cmap='RdYlGn')

        # fig2 = plt.figure(figsize=(10, 8))
        # ax2 = fig2.add_subplot(1, 1, 1)
        # sns.heatmap(df.corr(), vmin=-1, vmax=1, ax=ax2, linewidth=.5)


    @staticmethod
    def plot_data(image_data, text_emotion_data):

        all_labels = TextToEmotion.get_all_labels(text_emotion_data)
        all_ekman = TextToEmotion.get_all_ekman(text_emotion_data)

        x = np.array([s[2] for s in text_emotion_data])

        # cm = plt.get_cmap('gist_rainbow')
        cm = plt.get_cmap('Paired')
        fig = plt.figure(figsize=(12, 15))
        ax_text_all = fig.add_subplot(3, 1, 1)
        ax_text_ekman = fig.add_subplot(3, 1, 2)
        ax_img = fig.add_subplot(3, 1, 3)


        to_plot_general = []
        to_plot_ekman = []
        to_plot_img = []

        for label in all_labels:
            y = np.array(TextToEmotion.get_y_for_label(text_emotion_data, label))
            if np.quantile(y, 0.95) > 0.05 or np.max(y) > 0.2:
                to_plot_general.append((x, y, 'Text / ' + label))

        for ekman in all_ekman:
            y = np.array(TextToEmotion.get_y_for_ekman(text_emotion_data, ekman))
            to_plot_ekman.append((x, y, 'Text / ' + ekman))

        image_x, image_ys = Plotter.image_data_to_points(image_data)

        for y_label in image_ys:
            to_plot_img.append((image_x, image_ys[y_label], 'Face / ' + y_label))

        ax_text_all.set_prop_cycle('color', [cm(1.*i/len(to_plot_general))
                          for i in range(len(to_plot_general))])

        ax_text_ekman.set_prop_cycle(
            'color', [cm(1.*i/len(to_plot_ekman)) for i in range(len(to_plot_ekman))])

        ax_img.set_prop_cycle(
            'color', [cm(1.*i/len(to_plot_img)) for i in range(len(to_plot_img))])

        zipped_plots = np.array(to_plot_ekman + to_plot_img)
        # print(zipped_plots)
        Plotter.plot_covariance_matrix(zipped_plots)

        for (x, y, label) in to_plot_general:
            ax_text_all.plot(x, y, label=label)

        for (x, y, label) in to_plot_ekman:
            ax_text_ekman.plot(x, y, label=label)
        
        
        for (x, y, label) in to_plot_img:
            ax_img.plot(x, y, label=label)

        ax_text_all.set_xlabel('Minutes')
        ax_text_all.set_ylabel('Intensity')
        ax_text_all.set_title('Text to emotion plot')
        ax_text_all.legend(loc=(1.04, 0))

        ax_text_ekman.set_xlabel('Minutes')
        ax_text_ekman.set_ylabel('Intensity')
        ax_text_ekman.set_title('Text to ekman emotion plot')
        ax_text_ekman.legend(loc=(1.04, 0))

        ax_img.set_xlabel('Minutes')
        ax_img.set_ylabel('Intensity')
        ax_img.set_title('Image to emotion plot')
        ax_img.legend(loc=(1.04, 0))


    