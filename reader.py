# A script to implement a module of creating the classifier. The methods include getting all the files from the directory,
# and finding the word count for all the words in all the files in that directory.
import os

import nltk

# A function to get the files in a directory and store it in a list which is returned.
# @param __path is the path of the directory where the files have to be procured from.
def get_list_of_files(__path):
    __files = []
    for entry in os.listdir(__path):
        if os.path.isfile(os.path.join(__path, entry)):
            __files.append(__path + entry)
    return __files

# A function to get the word count of every word from a list of files which is stored in a dictionary that is returned.
# @param __files is list of files.
def get_word_count_from_files(__files):
    __dict = {}
    for file in __files:
        f = open(file, 'r')
        content = f.read()
        words = nltk.word_tokenize(content, 'english')
        for word in words:
            if len(word) > 2 or word.isalpha():
                if word in __dict:
                    __dict[word] += 1
                else:
                    __dict[word] = 1
    return remove_rare_words(__dict)

# A function to remove any word that has a count of lesser than 5.
# @param __dict is the dictionary of words with their count.
def remove_rare_words(__dict):
    __copy = __dict.copy()
    for word in __copy:
        if __copy[word] < 5:
            del __dict[word]
    return __dict



