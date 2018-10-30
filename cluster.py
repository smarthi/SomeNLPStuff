

import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from nltk.stem.snowball import SnowballStemmer

# TODO: This downloads some resources the first time it runs ...
#nltk.download('punkt')

# define set of stopwords

stopwords= {"title", "fee", "fees", "buyer", "seller", "for", "january", "february",
            "march", "may", "april", "june", "july", "august", "september", "october",
            "november", "december", "$", "jan", " feb", "mar", "apr", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec" , "of", "to", "and", "in", "if"}

synonym_dict = {
   "prep" : "preparation",
   "certify" : "certification",
   "doc" : "document",
   "trans" : "transaction"
}

stemmer = SnowballStemmer("english")

def tokenize(line):

    line = re.sub("[^a-zA-Z ]", " ", line).strip()

    tokens = []
    for token in nltk.word_tokenize(line):

        if not token.lower() in stopwords and len(token) > 1:
            if synonym_dict.keys().__contains__(token.lower()):
                token = synonym_dict.get(token.lower())

            tokens.append(stemmer.stem(token))

    if len(tokens) == 0:
        tokens.append("__ALL_TOKENS_REPLACED__")

    return tokens


def main():
    with open('raw_fee_fields.txt', 'r', encoding="utf-8") as code_file:
        fields = code_file.read().splitlines()

    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize, norm="l2", ngram_range=(1,3), analyzer='word')

    doc_matrix = vectorizer.fit_transform(fields)

    kmeans = KMeans(init='k-means++', n_clusters=30, n_init=10)

    kmeans.fit(doc_matrix)

    clusters = kmeans.labels_.tolist()

    class_fields = fields[-144:]
    d = dict()
    for i in clusters[-144:]:
        if i in d.keys():
            d.get(i).append(class_fields[i])
        else:
            d[i] = list([class_fields[i]])

    with open('fields_clustered.txt', 'w') as fields_clustered_file:
        for i in range(len(fields)):
            label = "outlier"
            if clusters[i] in d.keys():
                labels = d.get(clusters[i])
                if (len(labels)) > 0:
                    label = labels[0]

            fields_clustered_file.write(str(clusters[i]) + '\t' + fields[i] + '\t' + label + '\n' )

if __name__ == '__main__':
    main()
