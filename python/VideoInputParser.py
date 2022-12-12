import subprocess
from deepgram import Deepgram
import json
import re
import cv2
import matplotlib.pyplot as plt
import math
from ImagesEmotionLucas.scripts.frames_processing import ImageToEmotion
from plotter import Plotter
from text_to_emotion import TextToEmotion
from TextToContext import TextToContext
import tempfile
import os
from multiprocessing import Process

DEEPGRAM_API_KEY = '4d7ae42383509c771f0de389b766ffbb2f794ca8'
sentence_regex = re.compile(r'((?<=[.?!]")|((?<=[.?!])(?!")))\s*')


class VideoInputParser():
    def __init__(self):
        self.deepgram_client = Deepgram(DEEPGRAM_API_KEY)
        self.tte = TextToEmotion()
        self.ite = ImageToEmotion()
        self.ttc = TextToContext()

    def parse_video(self, video_bytes):
        with tempfile.NamedTemporaryFile(dir="temp_vids", delete=False, mode="w+b") as temp:
            print("Temp write")
            temp.write(video_bytes)
            temp.close()

            video_path = temp.name

            print("Temp write end at {}".format(video_path))
            return self.parse_video_from_path(video_path)

            # f = open('test_data/sample_response.json')
            # json_doc = json.load(f)

            # json_transcript = self._video_to_text(video_path)
            # # json_transcript = json_doc
            # sentences = self._process_transcript(json_transcript)
            # text_emotion_scores = self.tte.sentence_group_to_stats(sentences)
            # text_context_scores = self._detect_context(sentences)

            # images = self._extract_images_from_video(video_path)
            # parsed_images = self._parse_images(images)

            # data = Plotter.parse_data(
            #     parsed_images, text_emotion_scores, text_context_scores)

            # os.unlink(video_path)

            # return data

    def parse_video_from_path(self, video_path):
        f = open('test_data/sample_response.json')
        json_doc = json.load(f)
        json_transcript = json_doc

        # json_transcript = self._video_to_text(video_path)
        sentences = self._process_transcript(json_transcript)
        text_emotion_scores = self.tte.sentence_group_to_stats(sentences)
        text_context_scores = self._detect_context(sentences)

        images = self._extract_images_from_video(video_path)
        parsed_images = self._parse_images(images)

        data = Plotter.parse_data(
            parsed_images, text_emotion_scores, text_context_scores)

        os.unlink(video_path)

        return data

    def parse_and_plot_video(self, video_bytes):
        with tempfile.NamedTemporaryFile(dir="temp_vids", delete=False) as temp:
            temp.write(video_bytes)
            temp.close()

            video_path = temp.name

            f = open('test_data/sample_response.json')
            json_doc = json.load(f)

            images = self._extract_images_from_video(video_path)
            parsed_images = self._parse_images(images)

            # json_transcript = self._video_to_text(video_path)
            json_transcript = json_doc
            sentences = self._process_transcript(json_transcript)
            text_emotion_scores = self.tte.sentence_group_to_stats(sentences)
            text_context_scores = self._detect_context(sentences)

            Plotter.plot_data(
                parsed_images, text_emotion_scores, text_context_scores)

            os.unlink(video_path)

    def _detect_context(self, sentence_data):

        context = [self.ttc.predict(sentence)
                   for sentence, start, end in sentence_data]
        # for sentence, start, end in sentence_data:
        #     context = self.ttc.predict(sentence)
        #     print(context)

        x_offset = [(end+start)/2.0 for (_sentence, start, end)
                    in sentence_data]

        return list(zip(context, x_offset))

    def _extract_images_from_video(self, video_path, images_per_second=0.8):
        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps < 0.1:
            print("VIDEO NOT FOUND")
            return

        frames_to_skip = fps / images_per_second
        images = []
        stop = False
        current_frame = 0.0
        while not stop:
            video.set(cv2.CAP_PROP_POS_FRAMES, math.floor(current_frame))
            ret, img = video.read()
            if ret:
                timestamp = current_frame / fps
                images.append((img, timestamp))
                current_frame += frames_to_skip
            else:
                stop = True

        return images

    def _parse_images(self, images):
        images_parsed = []
        index = 0
        for img, timestamp in images:
            scores, max_label = self.ite.process_image(img)
            images_parsed.append(
                {'x': timestamp, 'y': scores, 'image_index': index})
            index += 1
            print("Parsed image {}/{}".format(index, len(images)))

        return images_parsed

    def _process_transcript(self, json):
        transcript_data = json['results']['channels'][0]['alternatives'][0]
        transcript = transcript_data['transcript']
        words = transcript_data['words']
        sentences = self._separate_sentences(transcript)
        sentences = self._timestamp_sentences(sentences, words)
        sentences = self._group_sentences(sentences)
        return sentences

    def _group_sentences(self, sentences, window_size=10):
        padding_start = [
            ('', sentences[0][1], sentences[0][1])] * (window_size-1)
        padding_end = [('', sentences[-1][2], sentences[-1][2])
                       ] * (window_size-1)
        sentences = padding_start + sentences + padding_end
        longer_sentences = []

        for i in range(len(sentences) - window_size + 1):
            sent_group = sentences[i:i + window_size]
            sentence = ' '.join([x[0] for x in sent_group])
            sentence_start = sent_group[0][1]
            sentence_end = sent_group[-1][2]
            longer_sentences.append(
                (sentence.strip(), sentence_start, sentence_end))

        return longer_sentences

    def _timestamp_sentences(self, sentences, words):
        word_index = 0
        sentences_timed = []
        for i in range(len(sentences)):
            sent = sentences[i]
            sentence_word_length = len(sent.split())
            word_range = words[word_index: (word_index+sentence_word_length)]
            time_start = word_range[0]['start']
            time_end = word_range[-1]['end']
            sentences_timed.append((sent, time_start, time_end))
            word_index += sentence_word_length

        return sentences_timed

    def _separate_sentences(self, text):
        sentences = [x for x in re.split(sentence_regex, text) if len(x) > 0]
        return sentences

    def _video_to_text(self, video_path):
        # Initializes the Deepgram SDK
        with open(video_path, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': 'audio/mp4'}
            options = {"punctuate": True, "model": "general",
                       "language": "en-US", "tier": "enhanced", "diarize": True}

            print('Requesting transcript...')
            response = self.deepgram_client.transcription.sync_prerecorded(
                source, options)
            # print(json.dumps(response, indent=4))
            print('Transcript obtained.')
            return response
