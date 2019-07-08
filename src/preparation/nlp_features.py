import gensim
import re

from spacy.lang.es import Spanish
from nltk import word_tokenize
from spacy.lemmatizer import Lemmatizer
from spacy.tokenizer import Tokenizer
from string import punctuation
from nltk.data import load
from nltk.stem import SnowballStemmer

nlp = Spanish()
tokenizer = Tokenizer(nlp.vocab)
lematizer = Lemmatizer()


class NLPFeatures(object):

    @staticmethod
    def stem_tokens(tokens, stemmer):
        """Pablo's work"""
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    @staticmethod
    def tokenize(text, stem=True, remove_stopwords=False):
        """Pablo's work"""
        stemmer = SnowballStemmer('spanish')
        spanish_tokenizer = load('tokenizers/punkt/spanish.pickle')

        # punctuation to remove
        non_words = list(punctuation)
        # we add spanish punctuation
        non_words.extend(['¿', '¡'])
        non_words.extend(map(str, range(10)))

        text = text.lower()

        result = []
        for sentence in spanish_tokenizer.tokenize(text):
            # remove punctuation
            text = ''.join([c for c in sentence if c not in non_words])
            # tokenize
            tokens = word_tokenize(text)

            # stem
            if stem:
                try:
                    stems = NLPFeatures.stem_tokens(tokens, stemmer)
                except Exception as e:
                    print(e)
                    print(text)
                    stems = ['']
                result += stems
            else:
                result += tokens

        return result

    @staticmethod
    def preprocess_mati(text, lemma=True):
        """Mati's work"""
        def is_trash(token):
            return token.is_punct or token.is_digit or \
                   token.is_stop or is_link(token)

        def is_link(token):
            return str(token).startswith("http")

        # def stem(word):
        #     p_stemmer = PorterStemmer()
        #     return p_stemmer.stem(word)

        def clean(text):
            text = (text
                    .lower()
                    .replace('\n', ' ')
                    .replace('\r', ''))
            tokens = text.split(" ")
            tokens = list(filter(lambda tk: not is_link(tk), tokens))
            text = ' '.join(tokens)
            non_words = list(punctuation)
            for each in non_words:
                text = text.replace(each, "")
            return text

        text = clean(text)
        tokens = tokenizer(text)
        tokens = list(filter(lambda tk: not is_trash(tk), tokens))
        if not lemma:
            return " ".join([tk.text.lower() for tk in tokens])
        return " ".join([tk.lemma_.lower() for tk in tokens])

    @staticmethod
    def preprocess_pablo(doc):
        """Pablo's work"""
        pre_doc = doc

        # remover URLs
        pre_doc = re.sub(
            r"https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ", pre_doc)

        # minúsculas
        pre_doc = pre_doc.lower()

        # volar acentos
        pre_doc = gensim.utils.deaccent(pre_doc)

        # remove bullshit
        pre_doc = re.sub(r"\@|\'|\"|\\|…|\/|\-|\||\(|\)|\.|\,|\!|\?|\:|\;|“|”|’|—", " ", pre_doc)

        # contraer vocales
        for v in 'aeiou':
            pre_doc = re.sub(r"[%s]+" % v, v, pre_doc)

            # normalizar espacio en blanco
        pre_doc = re.sub(r"\s+", " ", pre_doc)
        pre_doc = re.sub(r"(^\s)|(\s$)", "", pre_doc)

        return pre_doc
