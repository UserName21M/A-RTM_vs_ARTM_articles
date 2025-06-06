# %%

import json
import pickle

from collections import Counter

import pymorphy3
import string
import nltk

# %%

class TextProcessor:
    def __init__(self):
        self.stem2word_counter = {}

        self.stemmer = nltk.stem.PorterStemmer()
        self.stopwords_EN = set(self.stemmer.stem(i) for i in nltk.corpus.stopwords.words('english'))

        self.morph = pymorphy3.MorphAnalyzer()
        self.lemmatize = lambda str: self.morph.parse(str)[0].normal_form

        self.charfilter = set(list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789') + list(string.ascii_lowercase) + [' '])
        with open('stopwords-ru.txt', 'r', encoding = 'utf-8') as file:
            self.stopwords_RU = set(self.lemmatize(i.replace('\n', '')) for i in file.readlines())

    def preprocess_text_RU(self, x : str):
        x = x.lower().replace('\n', ' ')
        x = ''.join(i for i in x if i in self.charfilter)
        x = [self.lemmatize(i).replace('ё', 'е') for i in x.split(' ') if len(i) > 2 and len(i) < 20]
        return list(i for i in x if i not in self.stopwords_RU)

    def preprocess_text_EN(self, x : str):
        x = x.lower().replace('\n', ' ').replace('ё', 'е')
        x = ''.join(i for i in x if i in self.charfilter)
        x = [self.stem(i) for i in x.split(' ') if len(i) > 2 and len(i) < 20]
        return list(i for i in x if i not in self.stopwords_EN)

    def stem(self, word : str):
        stemmed = self.stemmer.stem(word)
        if stemmed not in self.stem2word_counter:
            self.stem2word_counter[stemmed] = Counter()
        self.stem2word_counter[stemmed].update([word])
        return stemmed

    def generate_stem2word(self):
        self.stem2word = {}
        for stem in self.stem2word_counter:
            self.stem2word[stem] = self.stem2word_counter[stem].most_common(1)[0][0]

        with open('stem2word', 'wb') as file:
            pickle.dump(self.stem2word, file) 

class DataGenerator:
    def __init__(self, data_path : str, processor : TextProcessor):
        self.data_path = data_path
        self.processor = processor
        with open(data_path, 'r', encoding = 'utf-8') as file:
            self.data = json.load(file)[1:]

        self.combinations = { }
        self.window = 3

    def generate(self):
        with open(self.data_path + '_data_vw', 'w', encoding = 'utf-8') as file:
            counter = Counter()
            lang = [article['language'] for article in self.data]
            lang = 'RU' if lang.count('RU') > lang.count('EN') else 'EN'

            docs = 0
            for article in self.data:
                if article['language'] != lang:
                    continue

                title, abstract = article['title'], article['abstract']
                if article['language'] == 'RU':
                    abstract = self.processor.preprocess_text_RU(abstract)
                    title = self.processor.preprocess_text_RU(title)
                else:
                    abstract = self.processor.preprocess_text_EN(abstract)
                    title = self.processor.preprocess_text_EN(title)

                ln = len(abstract)
                if ln < 30 or ln > 250:
                    continue

                counter.update(abstract)
                for _ in range(3):
                    counter.update(title)

                self.update_combinations(abstract)
                self.update_combinations(title)

                abstract = ' '.join(abstract)
                title = ' '.join(title)
                file.write('doc_%i |@default_class ' % docs + abstract + ' |@title_class ' + title + '\n')

                docs += 1
                if docs >= 249:
                    break

        with open(self.data_path + '_vocab_vw', 'w', encoding = 'utf-8') as file:
            file.writelines([w + '\n' for w,i in counter.items() if i >= 10])

        with open(self.data_path + '_combinations', 'wb') as file:
            pickle.dump(self.combinations, file) 

    def update_combinations(self, words : list):
        total = len(words)
        for pos1, word1 in enumerate(words):
            if word1 not in self.combinations:
                self.combinations[word1] = { }

            dct = self.combinations[word1]
            if not 'APPEARANCES' in dct:
                dct['APPEARANCES'] = 0
            dct['APPEARANCES'] += 1

            for pos2 in range(max(pos1 - self.window, 0), min(pos1 + self.window, total)):
                word2 = words[pos2]
                if word2 not in dct:
                    dct[word2] = 0
                dct[word2] += 1

# %%

import os

processor = TextProcessor()

for name in os.listdir('dataRU'):
    print('Working with', name)
    generator = DataGenerator('dataRU' + '/' + name, processor)
    generator.generate()

for name in os.listdir('dataEN'):
    print('Working with', name)
    generator = DataGenerator('dataEN' + '/' + name, processor)
    generator.generate()

processor.generate_stem2word()

# %%
