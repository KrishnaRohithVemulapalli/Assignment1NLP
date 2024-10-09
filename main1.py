from collections import Counter, defaultdict
from decimal import Decimal, getcontext

# Set decimal precision (adjust as needed)
getcontext().prec = 20  # You can set this to whatever level of precision you need


# Read corpus from file
def read_corpus(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip().lower().split()
            # Add </start> and </end> tokens to each sentence
            yield ['</start>'] + sentence + ['</end>']


# Count unigrams and bigrams
def count_ngrams(corpus):
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    total_unigrams = 0
    for sentence in corpus:
        # Update unigram counts
        unigram_counts.update(sentence)
        total_unigrams += len(sentence)
        # Update bigram counts
        for i in range(len(sentence) - 1):
            bigram_counts[sentence[i]][sentence[i + 1]] += 1
    return unigram_counts, bigram_counts, total_unigrams


# Compute unigram probabilities no smoothing
def compute_unigram_probs(unigram_counts, total_unigrams):
    unigram_probs = {}
    for word, count in unigram_counts.items():
        unigram_probs[word] = count / total_unigrams
    return unigram_probs


# Compute bigram probabilities no smoothing
def compute_bigram_probs(bigram_counts, unigram_counts):
    bigram_probs = defaultdict(dict)
    for w1, following_words in bigram_counts.items():
        for w2, count in following_words.items():
            bigram_probs[w1][w2] = count / unigram_counts[w1]
    return bigram_probs




# Export sorted results
def export_results(unigram_probs, bigram_probs, output_file):
    with open(output_file, 'w') as f:
        # Sort and write Unigram Probabilities (descending order)
        sorted_unigrams = sorted(unigram_probs.items(), key=lambda x: x[1], reverse=True)
        for word, prob in sorted_unigrams:
            f.write(f"{word}" + '\t' * 8 + f"{prob:.20f}\n")  # Decimal format with 20 digits precision
        # Sort and write Bigram Probabilities (descending order)
        for w1, following in bigram_probs.items():
            sorted_bigrams = sorted(following.items(), key=lambda x: x[1], reverse=True)
            for w2, prob in sorted_bigrams:
                f.write(f"{w1} {w2}" + '\t' * 7 + f"{prob:.20f}\n")  # Decimal format with 20 digits precision


# Main function to build and export the n-gram model
def build_ngram_model(train_file, output_file):
    # Read the corpus
    corpus = list(read_corpus(train_file))
    # Count unigrams and bigrams
    unigram_counts, bigram_counts, total_unigrams = count_ngrams(corpus)
    # Compute probabilities
    unigram_probs = compute_unigram_probs(unigram_counts, total_unigrams)
    bigram_probs = compute_bigram_probs(bigram_counts, unigram_counts)
    # Export results to a .txt file
    export_results(unigram_probs, bigram_probs, output_file)



train_file = 'train.txt'
output_file = 'ngram_probabilities.txt'
build_ngram_model(train_file, output_file)
