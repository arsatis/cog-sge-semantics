import os
import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# miscellaneous params
to_lemmatize = False


# global variables
dir = ""
train_set = None
processed_train_set = []


def get_dir():
    """
    Prepares directory for subsequent model saving.
    """
    dir = os.getcwd().split('/')
    if "models" not in dir:
        print("error: this file should be run from inside the `models` folder")
        exit(1)
    while dir[-1] != "models": dir.pop()
    return dir


def time_taken(func):
    """
    Decorator which calculates the time taken to perform each task.
    """
    def f(*args, **kwargs):
        print(f"starting {func.__name__}...")
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"total time taken to run {func.__name__}: {round(end - start, 2)}s")
    return f


@time_taken
def process_data():
    """
    Opens the SgE text files and processes them.
    """
    global train_set
    
    sgtb_text_dir = dir + ["raw_data", "SGTB_tagged", "utf8"]
    tagged_corpus = '/'.join(sgtb_text_dir)
    train_set = TaggedCorpusReader(
        root=tagged_corpus, 
        fileids=".*/.*\.txt",
        sep="_",
        sent_tokenizer=RegexpTokenizer("\r", gaps=True)
    )
    preprocess_input()


@time_taken
def preprocess_input():
    """
    Applies the standard NLP preprocessing steps on the text files.
    """
    global train_set, processed_train_set

    # remove stopwords, contractions, and punctuations
    remove_list = stopwords.words('english')
    contractions = ["'s", "'re", "'m", "'ve", "'d", "'ll", "'t", "n't", "na"]
    punctuations = [',', '.', '?', '!', '', ' ', ';', '--', '-', ':', '...', '....', '..',
                    '"', '``', "''", "'", "`", '-LRB-', '-RRB-', '@', '<p>']
    contracted_pronouns = ["u", "ur"]

    remove_list += contractions + punctuations + contracted_pronouns
    remove_set = set([w.lower() for w in remove_list])

    processed_train_set = [
        [
            word.lower().replace('ã¢â€â™', "'").replace('â€™', "'") \
            for (word, tag) in sent \
                if tag is not None and not word.startswith("*") and not word.lower() in remove_set
        ] for sent in train_set.tagged_sents()
    ]

    # lemmatization
    if to_lemmatize:
        print("starting lemmatization...")
        lemmatizer = WordNetLemmatizer()
        processed_train_set = [[lemmatizer.lemmatize(w) for w in sent] for sent in processed_train_set]


def train_models():
    """
    Trains the model on the processed training set.
    """
    sgtb_w2v_model = Word2Vec(
        processed_train_set,
        vector_size=100,
        window=5,
        min_count=10,
        epochs=10
    )
    sgtb_w2v_dir = dir + ["embeddings", "sgtb_word2vec.embedding"]
    sgtb_w2v_model.save('/'.join(sgtb_w2v_dir))

    # TODO: can try tfidf and lsi later


if __name__ == "__main__":
    dir = get_dir()
    process_data()
    print("total corpus size:", sum(map(len, processed_train_set))) # coca: 255382883, sgtb: 3390186
    train_models()
