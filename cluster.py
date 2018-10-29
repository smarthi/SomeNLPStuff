

import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# TODO: This downloads some resources the first time it runs ...
#nltk.download('punkt')

# define set of stopwords

stopwords= {"title", "fee", "fees", "buyer", "seller", "for", "january", "february",
            "march", "may", "april", "june", "july", "august", "september", "october",
            "november", "december", "$", "jan", " feb", "mar", "apr", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec" }

def tokenize(line):
    tokens = []
    for token in nltk.word_tokenize(line):
        token = re.sub("[^a-zA-Z ]", "", token.lower())

        if not token.lower() in stopwords and len(token) > 0:
            if not re.match("[^a-zA-Z]", token):
                tokens.append(token)

    if len(tokens) == 0:
        tokens.append("__ALL_TOKENS_REPLACED__")

    return tokens


def main():
    with open('raw_fee_fields.txt', 'r', encoding="utf-8") as code_file:
        fields = code_file.read().splitlines()

    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=tokenize, lowercase=True, norm="l2", stop_words=stopwords,
                                 ngram_range=(2,3), analyzer="word", token_pattern="[a-zA-Z]")

    doc_matrix = vectorizer.fit_transform(fields)

    kmeans = KMeans(init='k-means++', n_clusters=50, n_init=10)

    kmeans.fit(doc_matrix)

    clusters = kmeans.labels_.tolist()

    with open('fields_clustered.txt', 'w') as fields_clustered_file:
        for i in range(len(fields)):
            fields_clustered_file.write(str(clusters[i]) + '\t' + fields[i] + '\n')

if __name__ == '__main__':
    main()
