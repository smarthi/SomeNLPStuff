

import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from nltk.stem.snowball import SnowballStemmer

# TODO: This downloads some resources the first time it runs ...
#nltk.download('punkt')

# define set of stopwords

stopwords= {"fee", "fees", "buyer", "seller", "for", "january", "february",
            "march", "may", "april", "june", "july", "august", "september", "october",
            "november", "december", "$", "jan", " feb", "mar", "apr", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec" , "of", "to", "and", "in", "if"}

synonym_dict = {
   "prep" : "preparation",
   "certify" : "certification",
   "doc" : "document",
   "docs" : "document",
   "documentation" : "document",
   "trans" : "transaction",
   "cert" : "certification",
   "amend" : "amendment",
   "hoa" : "homeowners association",
   "condo": "condominium"
}

stemmer = SnowballStemmer("english")

def tokenize(line):

    line = re.sub("[^a-zA-Z ]", " ", line).strip().lower()

    tokens = []
    for token in nltk.word_tokenize(line):

        if not token in stopwords and len(token) > 1:
            if token in synonym_dict.keys():
                token = synonym_dict.get(token)

            for tok in nltk.word_tokenize(token):
                tokens.append(stemmer.stem(tok))

    if len(tokens) == 0:
        tokens.append("other")

    return tokens


def main():
    with open('raw_fee_fields.txt', 'r', encoding="utf-8") as code_file:
        fields = code_file.read().splitlines()

    vectorizer = TfidfVectorizer(use_idf=False, tokenizer=tokenize, norm="l2",  analyzer='word', sublinear_tf=True)

    doc_matrix = vectorizer.fit_transform(fields)

    kmeans = KMeans(init=doc_matrix[-144:].toarray(), n_clusters=144, n_init=10)

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
        for i in range(len(fields[:-144])):
            label = "other"
            if clusters[i] in d.keys():
                labels = d.get(clusters[i])
                if (len(labels)) > 0:
                    label = labels[0]

            fields_clustered_file.write(fields[i] + '\t' + str(clusters[i]) + '\t' + label + '\n' )

if __name__ == '__main__':
    main()
