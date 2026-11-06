"""Download the seven public test splits used in the paper into data/raw/ (Hugging Face hub; set HF_ENDPOINT to a
mirror if huggingface.co is unreachable)."""
import os, urllib.request
EP = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
FILES = {"imdb_test": "stanfordnlp/imdb/parquet/plain_text/test/0.parquet",
         "agnews_test": "fancyzhx/ag_news/parquet/default/test/0.parquet",
         "yelp_test": "Yelp/yelp_review_full/parquet/yelp_review_full/test/0.parquet",
         "tweet_offensive_test": "cardiffnlp/tweet_eval/parquet/offensive/test/0.parquet",
         "tweet_hate_test": "cardiffnlp/tweet_eval/parquet/hate/test/0.parquet",
         "emotion_test": "dair-ai/emotion/parquet/split/test/0.parquet",
         "amazon_test": "SetFit/amazon_reviews_multi_en/parquet/default/test/0.parquet"}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(out, exist_ok=True)
for name, path in FILES.items():
    dst = os.path.join(out, name + ".parquet")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"{EP}/api/datasets/{path}", dst); print("downloaded", name)
