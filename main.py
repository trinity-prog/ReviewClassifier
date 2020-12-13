import math

from reader import *


# A function to find the number of words from a file. Typically used for smoothing process.
# @param file_that_is_read is the file that is currently read and needed to find the number of words for.
def word_count_from_contents(file_that_is_read):
    result_dict = {}
    for word in file_that_is_read:
        result_dict[word] = 1
    return len(result_dict)


# Function to train the Bayes Classifier. Uses the two class directories and the test directory as parameters. The two
# class directories help create the training classifier and the test directory holds the data to be created with the
# classifier data.

def train_bayes(__path_pos, __path_neg, __path_testing):
    p_files = get_list_of_files(__path_pos)
    p_word_count = get_word_count_from_files(p_files)
    n_files = get_list_of_files(__path_neg)
    n_word_count = get_word_count_from_files(n_files)
    t_files = get_list_of_files(__path_testing)
    p_prob = {}
    n_prob = {}
    for file in t_files:
        fd = open(file)
        contents = nltk.word_tokenize(fd.read())
        for word in contents:
            # Laplace smoothing is implemented here. If the word is not present in the training data, to avoid
            # multiplying the entire probability by zero, the word is given a term frequency of 1. And the other
            # probabilities have 1 added to the numerator, and |V| which is the size of the distinct vocabulary of the
            # review.
            if len(word) > 1 or word.isalpha():
                if word in p_word_count:
                    p_prob[word] = (p_word_count[word] + 1) / (len(p_word_count) + word_count_from_contents(contents))
                else:
                    p_prob[word] = 1 / (len(p_word_count) + word_count_from_contents(contents))
                if word in n_word_count:
                    n_prob[word] = (n_word_count[word] + 1) / (len(n_word_count) + word_count_from_contents(contents))
                else:
                    n_prob[word] = 1 / (len(n_word_count) + word_count_from_contents(contents))
    return p_prob, n_prob


# Returns the probability of the word in that document.

def get_prob(prob_array, word):
    if word in prob_array:
        return prob_array[word]
    else:
        return 1


output_file = '/home/pree/sentimenalanalysis/IR/output/Test.txt'
outputf = open("Test", 'w+')


# A function to process the contents of the files in the test directory to check which class the file in the directory
# is likely to belong to. The probabilites are also normalised to avoid any error.

def process_contents(__contents, p_prob, n_prob):
    totalProbP = 1
    totalProbN = 1

    for a_word in __contents:
        if totalProbN < 1e-200 or totalProbP < 1e-200:
            totalProbN *= 1e200
            totalProbP *= 1e200
        totalProbP *= get_prob(p_prob, a_word)
        totalProbN *= get_prob(n_prob, a_word)
    if totalProbP > totalProbN:
        outputf.write(str(totalProbP) + ' | ' + str(totalProbN) + '\nReview is positive\n')
    else:
        outputf.write(str(totalProbP) + ' | ' + str(totalProbN) + '\nReview is negative\n')
    outputf.write("----------------------------------------------------------------------------------\n")


# A function that trains the model and uses the classifier to predict the likelihood of whether the review is positive
# or negative in the test directory.

def run():
    t_files = '/test/'
    pos_ = '/train/pos/'
    neg_ = '/train/neg/'
    p_prob, n_prob = train_bayes(__path_pos=pos_,
                                 __path_neg=neg_,
                                 __path_testing=t_files)
    test_list_files = get_list_of_files(t_files)
    for file in test_list_files:
        fd = open(file, 'r')
        outputf.write("Processing file: " + file + '\n')
        contents = nltk.word_tokenize(fd.read())
        process_contents(__contents=contents, p_prob=p_prob, n_prob=n_prob)


if __name__ == '__main__':
    run()





