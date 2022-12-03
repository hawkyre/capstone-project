import re
from text_to_emotion import TextToEmotion
import io
import unicodedata


class InputParser:

    def __init__(self, directory_path, wpm=130.0, script_name='script.txt'):
        # We expect a script.txt file and a series of photos named X.png, where X is a number inside the script file
        script_path = directory_path + '/' + script_name
        with io.open(script_path, 'rt', newline='') as f:
            script = f.read()
            script = unicodedata.normalize("NFKD", script)
            sentences = re.split(re.compile(
                '((?<=[.?!]")|((?<=[.?!])(?!")))\s*'), script)
            sent_imgs = [self.parse_sentence(s)
                         for s in sentences if s.strip() != '']
            self.input = sent_imgs

        self.wpm = wpm

    def parse_sentence(self, sentence):
        numbers = re.compile('\[\[(\d+)\]\]')
        matches = numbers.findall(sentence)
        # print('matches', matches, sentence)
        imgs = []
        sentence_len = len(sentence)
        for num in matches:
            img_name = "{}.png".format(num)
            chars_before_match = re.search(
                '\[\[{}\]\]'.format(num), sentence).span()[0]

            imgs.append((img_name, chars_before_match/sentence_len))

        clean_sentence = re.sub('\[\[\d+\]\]', "", sentence)
        clean_sentence = re.sub(re.compile('\s+'), " ", clean_sentence).strip()
        return {'sentence': clean_sentence, 'images': imgs}

    def group_sentences(self, sentences, size):

        longer_sentences = []
        for i in range(len(sentences) - size + 1):
            sent_group = '\n'.join(sentences[i:i + size - 1])
            start = len(' '.join(sentences[:i+1]).split(' ')) / self.wpm
            end = len(' '.join(sentences[:i + size]).split(' ')) / self.wpm
            longer_sentences.append((sent_group, start, end))

        return longer_sentences

    def parse(self):
        tte = TextToEmotion()
        sentences = [s['sentence'] for s in self.input]

        # longer_sentences = self.group_sentences(sentences, 4)
        # even_longer_sentences = self.group_sentences(sentences, 7)
        even_looonger_sentences = self.group_sentences(sentences, 12)
        # print(even_longer_sentences)

        text = '\n'.join([s['sentence'] for s in self.input])
        tte.text_to_graph(text)
        # tte.sentence_group_to_graph(longer_sentences)
        # tte.sentence_group_to_graph(even_longer_sentences)
        tte.sentence_group_to_graph(even_looonger_sentences)

        # print(sent_tte)

        # for sentence_data in self.input:
        #     print(sentence_data)
        #     sentence = sentence_data['sentence']
        #     images = sentence_data['images']
        #     sent_tte = tte.text_to_emotion(sentence)
        #     print(sent_tte)

        # Parse text to emotion
        # tte = ...
        # # Parse images to emotion
        # face_to_emotion = ...
        # # Parse text to context
        # ctx = ...
