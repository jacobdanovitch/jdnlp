import pandas as pd

from textacy.tm.topic_model import TopicModel
from textacy.vsm import Vectorizer

from jdnlp.utils.text import multi_replace
from nltk.corpus import stopwords

def series_to_docs(series, SUBS={}):
    docs = series.str.replace(r'[\.\,\?\-\+\(\)\[\]\*\#\!\$\%\:\d]+', ' ', regex=True).str.strip()
    # docs = docs.apply(lambda x: multi_replace(x, SUBS)) # 
    # docs = docs.str.replace(SUBS, ' ', regex=True)
    docs = docs.str.split(" ").values.tolist()
    docs = [[w.lower().strip() for w in words if w.strip() not in SUBS and len(w) > 2] for words in docs]
    return docs

def series_to_term_matrix(vectorizer, series, SUBS={}):
    docs = series_to_docs(series, SUBS)
    doc_term_matrix = vectorizer.transform(docs)
    return doc_term_matrix

if __name__ == '__main__':
    SUBS = {'PERSON', 'ORGANIZATION', 'GPE', 'in', 'to', 'he', 'at', 'of', 'as', 'it', 'is', 'an', 'on', 'be', ' ', 'his', 'up', 'if', "he's", 'the'}
    SUBS = {*SUBS, *{w for w in stopwords.words('english') if len(w) < 4}}
    # SUBS = {s: '' for s in SUBS}
    # SUBS = "|".join(f'(\b{s}\b)' for s in SUBS)

    # docs = pd.read_csv("datasets/twtc/preprocessed.csv", header=None)[0]
    df =  pd.read_csv("datasets/twtc/twtc.csv")
    docs = series_to_docs(df.report)


    vectorizer = Vectorizer(tf_type="linear", apply_idf=True, idf_type="smooth", norm="l2", min_df=3, max_df=0.95, max_n_terms=100000).fit(docs)
    model = TopicModel("nmf", n_topics=10)
    model.fit(vectorizer.transform(docs))

    for (name, pos) in {'of': ['OF'], 'inf': ['C', '1B', '2B', '3B'], 'p': ['LHP', 'RHP']}.items():
        docs = series_to_docs(df[df.primary_position.isin(pos)].report)
        doc_term_matrix = vectorizer.transform(docs)

        doc_topic_matrix = model.transform(doc_term_matrix)
        model.termite_plot(doc_term_matrix, vectorizer.id_to_term, save=f"twtc_termite_{name}.png", rank_terms_by='corpus_weight')