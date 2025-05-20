import pandas as pd
import re
import joblib
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(filename="preprocess.log", encoding="UTF-8"),
        logging.StreamHandler()
    ]
)

logging.info("Starting Preprocessing")

nltk.download('punkt')
nltk.download('stopwords')



#Load and sample dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.info("Faied to load dataset: %s", str(e))
    raise e

#Drop link column and preprocess
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

#Text Cleaning
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info(" Cleaning Text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("Text Cleaned")

#Vectorization
logging.info("Vectorization using TF-IDF..")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("TF-IDF matrix shape: %s", tfidf_matrix.shape)

#Cosine Similarity
logging.info("Calculating cosine similarity")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("Cosine Similarity Generated.")

#save everything
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("Data Saved to disk.")

logging.info("Preprocessing Complete.")




