import pandas as pd
from sklearn.model_selection import train_test_split

def index_and_split(df_fp, article_fp, test_size=0.15):
    print(df_fp, article_fp)
    df = pd.read_csv(df_fp)
    articles = pd.read_csv(article_fp)
    df = df.merge(articles, on="url")
    
    df = df.dropna()
    
    comment2i = dict(map(reversed, enumerate(df.body.unique())))
    text2i = dict(map(reversed, enumerate(df.text.unique())))
    
    df['comment_idx'] = df.body.apply(comment2i.get)
    df['text_idx'] = df.text.apply(text2i.get)
    
    df[["text", "text_idx"]].to_json("datasets/nnm/text_index.json")
    df[["body", "comment_idx"]].to_json("datasets/nnm/comment_index.json")
    df[["comment_idx", "text_idx"]].to_json("datasets/nnm/comment2article.json")
    
    train, test = train_test_split(df[["comment_idx", "text_idx"]], test_size=test_size, random_state=0)
    
    train.to_csv("datasets/nnm/train.csv", index=False)
    train.sample(7500).to_csv("datasets/nnm/train_sample.csv", index=False)
    
    test.to_csv("datasets/nnm/test.csv", index=False)
    test.sample(2500).to_csv("datasets/nnm/test_sample.csv", index=False)
    
    return True
    

if __name__ == '__main__':
    import sys
    print(sys.argv)
    index_and_split(*sys.argv[1:3])