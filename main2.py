from collections import Counter, defaultdict
from decimal import Decimal, getcontext

getcontext().prec = 20

def read_corpus(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip().lower().split()
            # Add </start> and </end> tokens to each sentence
            yield ['</start>'] + sentence + ['</end>']

# Replace infrequent words with <UNK>
def replace_with_unknown(corpus, threshold=1):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    return [
        [word if word_counts[word] > threshold else "<UNK>" for word in sentence]
        for sentence in corpus
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


# Compute unigram probabilities with smoothing
def compute_unigram_probs(unigram_counts, total_unigrams, vocab_size, k=1):
    unigram_probs = {}
    total_unigrams_decimal = Decimal(total_unigrams + k * vocab_size)
    for word, count in unigram_counts.items():
        unigram_probs[word] = Decimal(count + k) / total_unigrams_decimal
    return unigram_probs

# Compute bigram probabilities with smoothing
def compute_bigram_probs(bigram_counts, unigram_counts, vocab_size, k=1):
    bigram_probs = defaultdict(dict)
    for w1, following_words in bigram_counts.items():
        unigram_count_decimal = Decimal(unigram_counts[w1] + k * vocab_size)
        for w2, count in following_words.items():
            bigram_probs[w1][w2] = Decimal(count + k) / unigram_count_decimal
        # Ensure we account for unseen bigrams
        for unseen_word in unigram_counts.keys():
            if unseen_word not in following_words:
                bigram_probs[w1][unseen_word] = Decimal(k) / unigram_count_decimal
    return bigram_probs

# Update export_results to sort unigrams and bigrams by probabilities
def export_results(unigram_probs, bigram_probs, output_file):
    with open(output_file, 'w') as f:
        # Sort unigrams by probability in descending order
        sorted_unigrams = sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)
        for word, prob in sorted_unigrams:
            f.write(f"{word}" + '\t' * 8 + f"{prob:.20f}\n")
        # Sort bigrams by probability in descending order for each word
        for w1, following in bigram_probs.items():
            sorted_bigrams = sorted(following.items(), key=lambda x: x[1], reverse=True)
            for w2, prob in sorted_bigrams:
                f.write(f"{w1} {w2}" + '\t' * 7 + f"{prob:.20f}\n")

def build_ngram_model(train_file, output_file, k=1, threshold=1):
    corpus = list(read_corpus(train_file))
    corpus = replace_with_unknown(corpus, threshold=threshold)
    unigram_counts, bigram_counts, total_unigrams = count_ngrams(corpus)
    vocab_size = len(unigram_counts)
    unigram_probs = compute_unigram_probs(unigram_counts, total_unigrams, vocab_size, k=k)
    bigram_probs = compute_bigram_probs(bigram_counts, unigram_counts, vocab_size, k=k)
    export_results(unigram_probs, bigram_probs, output_file)



train_file = 'train.txt'
output_file = 'ngram_probabilities_Laplace_smoothing.txt'

# Example with Laplace (Add-1) smoothing
build_ngram_model(train_file, output_file, k=1, threshold=5)
# Example with Add-k smoothing (k=0.8)
build_ngram_model(train_file, 'ngram_probabilities_add_k_smoothing_1.txt', k=0.8, threshold=5)
# Example with Add-k smoothing (k=0.5)
build_ngram_model(train_file, 'ngram_probabilities_add_k_smoothing_2.txt', k=0.5, threshold=5)
