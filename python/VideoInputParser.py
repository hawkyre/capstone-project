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
import tempfile
import os

DEEPGRAM_API_KEY = '4d7ae42383509c771f0de389b766ffbb2f794ca8'
sentence_regex = re.compile(r'((?<=[.?!]")|((?<=[.?!])(?!")))\s*')


class VideoInputParser():
    def __init__(self):
        self.deepgram_client = Deepgram(DEEPGRAM_API_KEY)
        self.tte = TextToEmotion()
        self.ite = ImageToEmotion()

    def parse_video(self, video_bytes):
        with tempfile.NamedTemporaryFile(dir="temp_vids", delete=False) as temp:
            temp.write(video_bytes)
            temp.close()

            video_path = temp.name

            images = self._extract_images_from_video(video_path)
            parsed_images = self._parse_images(images)

            json_transcript = self._video_to_text(video_path)
            # json_transcript = json_doc
            sentences = self._process_transcript(json_transcript)
            text_emotion_scores = self.tte.sentence_group_to_stats(sentences)

            Plotter.plot_data(parsed_images, text_emotion_scores)

            os.unlink(video_path)

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
            print("Parsed image {}".format(index))
            index += 1

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
            print('Your file may take up to a couple minutes to process.')
            print(
                'While you wait, did you know that Deepgram accepts over 40 audio file formats? Even MP4s.')
            print(
                'To learn more about customizing your transcripts check out developers.deepgram.com')

            response = self.deepgram_client.transcription.sync_prerecorded(source, options)
            # print(json.dumps(response, indent=4))
            print('Transcript obtained.')
            return response
