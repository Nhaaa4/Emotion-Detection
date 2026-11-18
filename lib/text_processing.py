import contractions
import emoji
import re
import unicodedata

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TweetTokenizer


SKIN_TONE = re.compile(r"_(?:light|medium-light|medium|medium-dark|dark)_skin_tone")
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor", "never", "up", "down"}

tweet_tokenizer  = TweetTokenizer()
lemmatizer = WordNetLemmatizer()


def clean_text(text, keep_hashtags=True, keep_emoji=True):
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#" if keep_hashtags else r"#\w+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    if keep_emoji:
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = SKIN_TONE.sub("", text)
        text = text.replace("_", " ")
    else:
        text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"[^A-Za-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)               
    text = text.replace("\r\n", "\n") 
    text = text.casefold()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) 
    text = text.split()
    text = [contractions.fix(t) for t in text]
    text = " ".join([t for t in text])
    tokens = tweet_tokenizer.tokenize(text)
    return tokens


def remove_stopwords(tokens):
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]


def to_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word, to_wordnet_pos(tag)) for word, tag in pos_tag(tokens)]


def text_preprocessing(text, stopwords=True, lemmatize=True, hashtags=True, emoji=True):
    text = clean_text(text, keep_hashtags=hashtags, keep_emoji=emoji)
    tokens = normalize_text(text)
    if stopwords: 
        tokens = remove_stopwords(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)   
    return " ".join(tokens)