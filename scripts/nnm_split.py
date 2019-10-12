import sys
sys.path.append('../jdnlp')

from jdnlp.data.indexing import index_and_split

df_fp, article_fp = "~/COMP5900/NRM/reddit_100000.csv", "~/COMP5900/NRM/article_texts.csv"
index_and_split(df_fp, article_fp)