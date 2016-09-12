import string
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = './hw1datasets'
token_dict = {}
all_stemmed_words = []
ps = PorterStemmer()
stop_words = set(stopwords.words("english")) | set(string.punctuation)
output_file = open('hw1output.txt', 'w')
text_lines = "-----------------------------------------------------\n"

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        file_contents = open(file_path, 'r')
        text = file_contents.read()
        lowered = text.lower()
        token_dict[file] = lowered
        file_contents.close()

num_docs = len(token_dict)

doc_names = []
for file_name in token_dict.keys():
    doc_names.append(file_name)

for file in token_dict.keys():
    words = word_tokenize(token_dict[file])

    output_file.write(text_lines + "Sentence tokenizing " + file + ".\n" + text_lines)
    output_file.write(str(sent_tokenize(token_dict[file])) + "\n")

    output_file.write(text_lines + "Word tokenizing " + file + ".\n" + text_lines)
    output_file.write(str(words) + "\n")

    no_stop_words = []
    for w in words:
        if w not in stop_words:
            no_stop_words.append(w)
    output_file.write(text_lines + "Stop words removed from " + file + ".\n" + text_lines)
    output_file.write(str(no_stop_words) + "\n")

    stemmed_words = []
    for w in words:
        if w not in stop_words:
            stemmed_words.append(ps.stem(w))
    output_file.write(text_lines + "Stemming " + file + ".\n" + text_lines)
    output_file.write(str(stemmed_words) + "\n")

    all_stemmed_words.append(stemmed_words)

tfidf = TfidfVectorizer(all_stemmed_words, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
doc_matrix = tfs.toarray()
set_vocab = tfidf.get_feature_names()

output_file.write(text_lines + "TF-IDF document-word matrix.\n" + text_lines)
output_file.write("%-15s%-20s%-20s%-20s%-20s%-20s\n" % (
    "", str(doc_names[0]), str(doc_names[1]), str(doc_names[2]), str(doc_names[3]), str(doc_names[4])))
for i in range(0, len(set_vocab)):
    output_file.write("%-15s%-20s%-20s%-20s%-20s%-20s\n" % (
        str(set_vocab[i]), str(doc_matrix[0][i]), str(doc_matrix[1][i]),
        str(doc_matrix[2][i]), str(doc_matrix[3][i]), str(doc_matrix[4][i])))

output_file.write(text_lines + "Cosine similarity.\n" + text_lines)
for i in range(0, num_docs):
    for j in range(i, num_docs):
        if i != j:
            output_file.write("Cosine similarity of %s to %s is %s. \n" % (
            doc_names[i], doc_names[j], cosine_similarity(tfs[i,], tfs[j,])))

output_file.close()