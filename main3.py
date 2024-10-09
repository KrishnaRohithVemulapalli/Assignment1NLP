from collections import Counter, defaultdict
from decimal import Decimal, getcontext
import math

getcontext().prec = 20  # Set precision

def read_corpus(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip().lower().split()
            # Add </start> and </end> tokens to each sentence
            yield ['</start>'] + sentence + ['</end>']

# Function to replace rare words in the training corpus with <UNK> based on frequency
def replace_with_unknown(corpus, threshold=1):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    return [
        [word if word_counts[word] > threshold else "<UNK>" for word in sentence]
        for sentence in corpus
    ]

# Function to replace words in the validation corpus that are not in the training corpus vocabulary
def replace_with_unk(val_corpus, training_vocab):
    return [
        [word if word in training_vocab else "<UNK>" for word in sentence]
        for sentence in val_corpus
    ]

def count_ngrams(corpus):
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    total_unigrams = 0
    for sentence in corpus:
        unigram_counts.update(sentence)
        total_unigrams += len(sentence)
        for i in range(len(sentence) - 1):
            bigram_counts[sentence[i]][sentence[i + 1]] += 1
    return unigram_counts, bigram_counts, total_unigrams

def compute_unigram_probs(unigram_counts, total_unigrams, vocab_size, k=1):
    unigram_probs = {}
    total_unigrams_decimal = Decimal(total_unigrams + k * vocab_size)
    for word, count in unigram_counts.items():
        unigram_probs[word] = Decimal(count + k) / total_unigrams_decimal
    return unigram_probs

def compute_bigram_probs(bigram_counts, unigram_counts, vocab_size, k=1):
    bigram_probs = defaultdict(dict)
    for w1, following_words in bigram_counts.items():
        unigram_count_decimal = Decimal(unigram_counts[w1] + k * vocab_size)
        for w2, count in following_words.items():
            bigram_probs[w1][w2] = Decimal(count + k) / unigram_count_decimal
        # Account for unseen bigrams
        for unseen_word in unigram_counts.keys():
            if unseen_word not in following_words:
                bigram_probs[w1][unseen_word] = Decimal(0 + k) / unigram_count_decimal
    return bigram_probs

def compute_perplexity(corpus, unigram_probs, bigram_probs, N, ngram_size=2):
    log_prob_sum = Decimal(0)
    for sentence in corpus:
        for i in range(ngram_size - 1, len(sentence)):
            if ngram_size == 1:
                word = sentence[i]
                prob = unigram_probs.get(word)
            elif ngram_size == 2:
                w1, w2 = sentence[i - 1], sentence[i]
                prob = bigram_probs.get(w1).get(w2)
            else:
                raise NotImplementedError("Only unigram and bigram models are supported for now.")
            log_prob_sum += Decimal(math.log(prob))

    avg_log_prob = log_prob_sum / Decimal(N)
    perplexity = Decimal(math.exp(-avg_log_prob))

    return perplexity


def build_ngram_model(train_file, output_file, val_file=None, k=1, threshold=1, ngram_size=2):
    # Read and preprocess the training corpus
    corpus = list(read_corpus(train_file))
    corpus = replace_with_unknown(corpus, threshold=threshold)

    # Count unigrams and bigrams
    unigram_counts, bigram_counts, total_unigrams = count_ngrams(corpus)
    vocab_size = len(unigram_counts)

    # Compute probabilities
    unigram_probs = compute_unigram_probs(unigram_counts, total_unigrams, vocab_size, k=k)
    bigram_probs = compute_bigram_probs(bigram_counts, unigram_counts, vocab_size, k=k)



    # Read and preprocess the validation corpus
    val_corpus = list(read_corpus(val_file))
    training_vocab = set(unigram_counts.keys())
    val_corpus = replace_with_unk(val_corpus, training_vocab)

    # Compute perplexity for both training and validation data
    train_N = sum(len(sentence) for sentence in corpus) + k * vocab_size
    perplexity = compute_perplexity(corpus, unigram_probs, bigram_probs, N=train_N, ngram_size=1)
    print(f"\n\n\nPerplexity of Unigram Model on Training data using {'Laplace' if k==1 else 'Add-'+str(k)} smoothing: {perplexity:.20f}")

    val_N = sum(len(sentence) for sentence in val_corpus)
    perplexity = compute_perplexity(val_corpus, unigram_probs, bigram_probs, N=val_N, ngram_size=1)
    print(f"Perplexity of Unigram Model on Validation data using {'Laplace' if k==1 else 'Add-'+str(k)} smoothing: {perplexity:.20f}")

    train_N = sum(len(sentence) for sentence in corpus) + k * vocab_size
    perplexity = compute_perplexity(corpus, unigram_probs, bigram_probs, N=train_N, ngram_size=ngram_size)
    print(f"Perplexity of Bi-gram Model on Training data using {'Laplace' if k==1 else 'Add-'+str(k)} smoothing: {perplexity:.20f}")

    val_N = sum(len(sentence) for sentence in val_corpus)
    perplexity = compute_perplexity(val_corpus, unigram_probs, bigram_probs, N=val_N, ngram_size=ngram_size)
    print(f"Perplexity of Bi-gram Model on Validation data using {'Laplace' if k==1 else 'Add-'+str(k)} smoothing: {perplexity:.20f}")

# Example usage
train_file = 'train.txt'
val_file = 'val.txt'
output_file = 'ngram_probabilities_smoothing.txt'

# Laplace
build_ngram_model(train_file, output_file, val_file=val_file, k=1, threshold=5, ngram_size=2)

build_ngram_model(train_file, 'ngram_probabilities_add_k.txt', val_file=val_file, k=0.8, threshold=5, ngram_size=2)
# Add K
build_ngram_model(train_file, 'ngram_probabilities_add_k.txt', val_file=val_file, k=0.5, threshold=5, ngram_size=2)

