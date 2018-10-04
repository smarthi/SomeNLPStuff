

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# TODO: This downloads some resources the first time it runs ...
#nltk.download('punkt')

def tokenize(line):
    # TODO:stemming, stop word
    return nltk.word_tokenize(line)

def main():
    with open('raw_fee_fields.txt', 'r', encoding="utf-8") as code_file:
        fields = code_file.read().splitlines()

    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize)

    doc_matrix = vectorizer.fit_transform(fields)

    kmeans = KMeans(n_clusters=100)

    kmeans.fit(doc_matrix)

    clusters = kmeans.labels_.tolist()

    with open('fields_clustered.txt', 'w') as fields_clustered_file:
        for i in range(len(fields)):
            fields_clustered_file.write(fields[i] + '\t' + str(clusters[i]) + '\n')

if __name__ == '__main__':
    main()
