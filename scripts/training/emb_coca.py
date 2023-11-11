import os
import random
import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


# miscellaneous params
is_test = False
to_lemmatize = False


# global variables
dir = ""
train_set = []
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
    Opens a subset of COCA text files and processes them.
    """
    coca_text_dir = dir + ["raw_data", "COCA_text"]
    genres = os.listdir('/'.join(coca_text_dir))

    for genre in genres:
        # exclude certain genres to reduce training sample size
        if genre in {'text_acad_isi', 'text_fic_jjw', 'text_mag_jgr', 'text_news_nne', 'text_tvm_nwh'}:
            continue

        coca_text_dir.append(genre)
        files = os.listdir('/'.join(coca_text_dir))

        for file in files:
            coca_text_dir.append(file)
            with open('/'.join(coca_text_dir), 'r') as f:
                corpus = f.read()
                corpus_lines = corpus.split('\n')
            coca_text_dir.pop()

            for line in corpus_lines:
                train_set.append(line)

        coca_text_dir.pop()
        sample_train_subset()
        preprocess_input()
        print(f"done working on {genre}")

        if is_test:
            return


@time_taken
def sample_train_subset():
    global train_set

    # randomly sample 5% of the training set (5% is just a heuristic)
    n = len(train_set)
    sampled_train_set = random.sample(train_set, n * 5 // 100)
    train_set = sampled_train_set


@time_taken
def preprocess_input():
    """
    Applies the standard NLP preprocessing steps on the text files.
    This is done for each genre rather than for the entire input at one shot to avoid overwhelming RAM.
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

    train_set = [[w.lower() for w in sent.split() if w.lower() not in remove_set] for sent in train_set]

    # lemmatization
    if to_lemmatize:
        print("starting lemmatization...")
        lemmatizer = WordNetLemmatizer()
        train_set = [[lemmatizer.lemmatize(w) for w in sent] for sent in train_set]
    
    if is_test:
        print(train_set[:2])
    
    processed_train_set += train_set
    train_set.clear()


def train_models():
    """
    Trains the model on the processed training set.
    """
    coca_w2v_model = Word2Vec(
        processed_train_set,
        vector_size=100,
        window=5,
        min_count=10,
        epochs=10
    )
    coca_w2v_dir = dir + ["embeddings", "coca_word2vec.embedding"]
    coca_w2v_model.save('/'.join(coca_w2v_dir))

    # TODO: can try tfidf and lsi later


if __name__ == "__main__":
    dir = get_dir()
    process_data()
    print("total corpus size:", sum(map(len, processed_train_set))) # coca: 255382883, sgtb: 3390186
    if not is_test:
        train_models()
