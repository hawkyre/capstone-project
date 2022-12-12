from text_to_emotion import TextToEmotion
import numpy as np
import matplotlib.pyplot as plt
import copy 
import math 
import seaborn as sns
import pandas as pd
from utils import normalize_x
from time_series_detection import detect_significant_trends

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

        labels = list(map(lambda point: point['label'], data))
        y = list(map(lambda point: point['y'], data))

        # ax.set_title('Ekman normalized')
        # ax.plot()
        df = pd.DataFrame(np.array(y).T, columns=labels)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        image_text_corr_mat = df.corr().to_numpy()[6:12, :6]

        sns.heatmap(image_text_corr_mat, vmin=-1, vmax=1, ax=ax, linewidth=.5, xticklabels=df.columns[:6], yticklabels=df.columns[6:12], cmap='RdYlGn')

        # fig2 = plt.figure(figsize=(10, 8))
        # ax2 = fig2.add_subplot(1, 1, 1)
        # sns.heatmap(df.corr(), vmin=-1, vmax=1, ax=ax2, linewidth=.5)

    @staticmethod
    def create_correlation_matrix(rows, cols, title, selector):
        
        sample = cols + rows

        labels = list(map(lambda sample: sample['label'], sample))
        y = list(map(lambda sample: sample['y'], sample))

        df = pd.DataFrame(np.array(y).T, columns=labels)
                
        corr_mat = df.corr().to_numpy()[len(cols):, :len(cols)]
        print("Corre mat:\n {}".format(corr_mat))

        return {
            'selector': selector,
            'title': title,
            'corr_mat': corr_mat.tolist(),
            'row_labels': list(map(lambda r: r['label'], rows)),
            'col_labels': list(map(lambda r: r['label'], cols))
        }

        # sns.heatmap(image_text_corr_mat, vmin=-1, vmax=1, ax=ax, linewidth=.5, xticklabels=df.columns[:6], yticklabels=df.columns[6:12], cmap='RdYlGn')

        # fig2 = plt.figure(figsize=(10, 8))
        # ax2 = fig2.add_subplot(1, 1, 1)
        # sns.heatmap(df.corr(), vmin=-1, vmax=1, ax=ax2, linewidth=.5)

    @staticmethod
    def normalize_image_data(image_data):
        image_data_normalized = []

        kernel = np.array([1, 2, 5, 10, 20, 30, 20, 10, 5, 2, 1])
        kernel = kernel / np.sum(kernel)
        kernel = kernel * 1.25
        
        image_x, image_ys = Plotter.image_data_to_points(image_data)

        for y_label in image_ys:
            x = image_x
            y = image_ys[y_label]
            y = np.convolve(y, kernel, 'same')
            x, y = normalize_x(x, y)
            

            info_type = 'face'
            subtopic = y_label
            label = 'Face / ' + y_label

            point = {
                'x': x.tolist(),
                'y': y.tolist(),
                'info_type': info_type,
                'subtopic': subtopic,
                'label': label
            }

            image_data_normalized.append(point)
        
        return image_data_normalized

    @staticmethod
    def normalize_context_data(context_data):
        context_data_normalized = []
        all_labels = [x[0] for x in context_data[0][0]]

        context_map = {}
        x_series = []

        for context_values, x in context_data:
            x_series.append(x)
            for context, score in context_values:
                if not context_map.get(context):
                    context_map.setdefault(context, [score])
                else:
                    context_map[context].append(score)
        
        for label in all_labels:
            y_series = context_map[label]
            x, y = normalize_x(x_series, y_series)

            info_type = 'context'
            subtopic = label
            label = 'Context / ' + label

            point = {
                'x': x.tolist(),
                'y': y.tolist(),
                'info_type': info_type,
                'subtopic': subtopic,
                'label': label
            }

            context_data_normalized.append(point)
        
        context_data_normalized = sorted(context_data_normalized, key=lambda x: np.sum(np.array(x['y'])**2), reverse=True)
        return context_data_normalized[:8]

    @staticmethod
    def normalize_text_emotion_data(text_emotion_data):
        text_emotion_data_normalized = []
        
        all_labels = TextToEmotion.get_all_labels(text_emotion_data)
        all_ekman = TextToEmotion.get_all_ekman(text_emotion_data)
        
        x = np.array([s[2] for s in text_emotion_data])

        for label in all_labels:
            y = np.array(TextToEmotion.get_y_for_label(text_emotion_data, label))

            info_type = 'text'
            subtopic = label
            label = 'Text emotion / ' + label


            if np.quantile(y, 0.95) > 0.05 or np.max(y) > 0.2:
                x_norm, y_norm = normalize_x(x, y)
                point = {
                    'x': x_norm.tolist(),
                    'y': y_norm.tolist(),
                    'info_type': info_type,
                    'subtopic': subtopic,
                    'label': label
                }
                text_emotion_data_normalized.append(point)

        for ekman_label in all_ekman:
            y = np.array(TextToEmotion.get_y_for_ekman(text_emotion_data, ekman_label))

            info_type = 'text_ekman'
            subtopic = ekman_label
            ekman_label = 'Text emotion (Ekman) / ' + ekman_label


            if np.quantile(y, 0.95) > 0.05 or np.max(y) > 0.2:
                x_norm, y_norm = normalize_x(x, y)
                point = {
                    'x': x_norm.tolist(),
                    'y': y_norm.tolist(),
                    'info_type': info_type,
                    'subtopic': subtopic,
                    'label': ekman_label
                }
                text_emotion_data_normalized.append(point)
        
        return text_emotion_data_normalized

    @staticmethod
    def draw_plot(points, title):
        cm = plt.get_cmap('Paired')
        fig = plt.figure(figsize=(15, 4))
        axis = fig.add_subplot(1, 1, 1)

        axis.set_prop_cycle('color', [cm(1.*i/len(points))
                          for i in range(len(points))])

        for point in points:
            x = point['x']
            y = point['y']
            label = point['label']
            axis.plot(x, y, label=label)

        axis.set_xlabel('Minutes')
        axis.set_ylabel('Intensity')
        axis.set_title(title)
        axis.legend(loc=(1.04, 0))

    @staticmethod
    def parse_data(image_data, text_emotion_data, text_context_scores):
        text_all_points = Plotter.normalize_text_emotion_data(text_emotion_data)
        img_points = Plotter.normalize_image_data(image_data)
        context_points = Plotter.normalize_context_data(text_context_scores)

        ekman_data = list(filter(lambda x: x['info_type'] == 'text_ekman', text_all_points))

        # happiness_trend = detect_significant_trends(happy['y'], happy['x'][1] - happy['x'][0])
        # print('happiness trend: ', happiness_trend)

        all_data = text_all_points + img_points + context_points
        all_trends = {}

        for ekman_sample in ekman_data:
            print('-'*30)
            print(ekman_sample['label'])
            trends = detect_significant_trends(ekman_sample['y'], ekman_sample['x'][1] - ekman_sample['x'][0])
            all_trends.setdefault(ekman_sample['label'], trends)
        
        text_img_corr_map = Plotter.create_correlation_matrix(ekman_data, img_points, "Correlation between text emotion and facial emotion.", "Text / Face")
        text_context_corr_map = Plotter.create_correlation_matrix(ekman_data, context_points, "Correlation between text emotion and context.", "Text / Context")
        img_context_corr_map = Plotter.create_correlation_matrix(img_points, context_points, "Correlation between facial emotion and context.", "Face / Context")

        return { 'time_series': all_data, 'trends': all_trends, 'correlations': [text_img_corr_map, text_context_corr_map, img_context_corr_map] }

    @staticmethod
    def plot_data(image_data, text_emotion_data, text_context_scores):
        text_all_points = Plotter.normalize_text_emotion_data(text_emotion_data)
        text_general_points = list(filter(lambda x: x['info_type'] == 'text', text_all_points))
        text_ekman_points = list(filter(lambda x: x['info_type'] == 'text_ekman', text_all_points))
        
        img_points = Plotter.normalize_image_data(image_data)
        
        context_points = Plotter.normalize_context_data(text_context_scores)

        Plotter.draw_plot(text_general_points, 'Text to Emotion time series')
        Plotter.draw_plot(text_ekman_points, 'Text to Ekman Emotion time series')
        Plotter.draw_plot(img_points, 'Image to Emotion time series')
        Plotter.draw_plot(context_points, 'Text to Context time series')

        # ax_text_all.set_prop_cycle('color', [cm(1.*i/len(text_general_points))
        #                   for i in range(len(text_general_points))])

        # ax_text_ekman.set_prop_cycle(
        #     'color', [cm(1.*i/len(text_ekman_points)) for i in range(len(text_ekman_points))])

        # ax_img.set_prop_cycle(
        #     'color', [cm(1.*i/len(img_points)) for i in range(len(img_points))])

        # ax_ctx.set_prop_cycle(
        #     'color', [cm(1.*i/len(img_points)) for i in range(len(img_points))])

        zipped_plots = np.array(text_ekman_points + img_points)
        Plotter.plot_covariance_matrix(zipped_plots)

        # for point in text_general_points:
        #     x = point['x']
        #     y = point['y']
        #     label = point['label']
        #     ax_text_all.plot(x, y, label=label)

        # for point in text_ekman_points:
        #     x = point['x']
        #     y = point['y']
        #     label = point['label']
        #     ax_text_ekman.plot(x, y, label=label)        
        
        # for point in img_points:
        #     x = point['x']
        #     y = point['y']
        #     label = point['label']
        #     ax_img.plot(x, y, label=label)

        # ax_text_all.set_xlabel('Minutes')
        # ax_text_all.set_ylabel('Intensity')
        # ax_text_all.set_title('Text to emotion plot')
        # ax_text_all.legend(loc=(1.04, 0))

        # ax_text_ekman.set_xlabel('Minutes')
        # ax_text_ekman.set_ylabel('Intensity')
        # ax_text_ekman.set_title('Text to ekman emotion plot')
        # ax_text_ekman.legend(loc=(1.04, 0))

        # ax_img.set_xlabel('Minutes')
        # ax_img.set_ylabel('Intensity')
        # ax_img.set_title('Image to emotion plot')
        # ax_img.legend(loc=(1.04, 0))


    