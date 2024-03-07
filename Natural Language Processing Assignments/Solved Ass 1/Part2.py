
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
import math
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:

                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token

class LanguageModel:
    def __init__(self, corpus):
        print("")

    def train(self, corpus):
        raise NotImplementedError("Train method must be implemented in subclasses")

    def prob(self, words):
        raise NotImplementedError("Probability method must be implemented in subclasses")

    def draw(self, prev_word):
        raise NotImplementedError("Draw method must be implemented in subclasses")

    def generateSentence(self):
        sentence = [start]
        while True:
            prev_word = sentence[-1]
            next_word = self.draw(prev_word)
            sentence.append(next_word)
            if next_word == end or len(sentence) > 20:
                break
        return sentence

    def getSentenceProbability(self, sen):
        log_probability = 0.0
        for i in range(1, len(sen)):
            word = sen[i]
            prev_word = sen[i - 1]
            word_prob = self.prob((prev_word, word))
            if word_prob == 0:
                return float('-inf')
            log_probability += math.log(word_prob)
        return log_probability

    def getCorpusPerplexity(self, corpus):
        total_log_probability = 0.0
        total_words = 0

        for sentence in corpus:
            total_words += len(sentence)
            total_log_probability += self.getSentenceProbability(sentence)

        perplexity = math.exp(-total_log_probability / total_words)
        return perplexity

    def generateSentencesToFile(self, number_of_sentences, filename):
        file_pointer = open(filename, 'w+')
        for _ in range(number_of_sentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            string_generated = f"{prob} {' '.join(sen)}"
            print(string_generated, end="\n", file=file_pointer)
        file_pointer.close()


class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0

    def prob(self, words):
        word = words[1]
        return self.counts[word] / self.total

    def draw(self, prev_word):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob((prev_word, word))
            if rand <= 0.0:
                return word


class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
        self.vocab_size = len(self.counts)

    def prob(self, words):
        word = words[1]
        return (self.counts[word] + 1) / (self.total + self.vocab_size)

    def draw(self, prev_word):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob((prev_word, word))
            if rand <= 0.0:
                return word

# The main routine
if __name__ == "__main__":
    train_corpus = readFileToCorpus('train.txt')
    train_corpus = preprocess(train_corpus)

    pos_test_corpus = readFileToCorpus('pos_test.txt')
    neg_test_corpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    for sentence in train_corpus:
        for word in sentence:
            vocab.add(word)

    pos_test_corpus = preprocessTest(vocab, pos_test_corpus)
    neg_test_corpus = preprocessTest(vocab, neg_test_corpus)

    pos_test = UnigramModel(train_corpus)
    pos_test.generateSentencesToFile(20, "Unigram_Output.txt")
