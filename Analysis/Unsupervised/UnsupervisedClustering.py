import json
import os
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GANDHI_PATH = os.path.join(SCRIPT_DIR, "Datasets", "The-Collected-Works-Mahatma-Gandhi.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("1. SETUP, STOPWORDS & DATA LOADING")
print("=" * 70)

# Define custom stopwords based on dataset specifics
custom_stop_words = [
    "bapu", "blessings", "chi", "dear", "mr", "sincerely", "gandhi",
    "letter", "people", "time", "ashram", "sabarmati", "sevagram", "glad",
    "know", "shall", "indian", "passage", "wardha", "segaon", "wish", "quite",
    "sir", "yours", "mahatma", "january", "february", "ba", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "bhai", "friend", "vandemataram", "improve",
    "mohandas", "sjt", "thanks", "health", "good", "reply", "able",
    "hope", "letters", "shri", "today", "thank", "copy", "esq", "long",
    "received", "happy", "dr", "right", "day", "long", "just"
]

# A large list of common English verbs in various forms to prevent them from dominating topics
common_verbs = [
    "say", "said", "says", "saying", "go", "went", "gone", "goes", "going", 
    "make", "made", "makes", "making", "take", "took", "taken", "takes", "taking", 
    "get", "got", "gotten", "gets", "getting", "think", "thought", "thinks", "thinking", 
    "see", "saw", "seen", "sees", "seeing", "come", "came", "comes", "coming", 
    "want", "wanted", "wants", "wanting", "look", "looked", "looks", "looking", 
    "use", "used", "uses", "using", "find", "found", "finds", "finding", 
    "give", "gave", "given", "gives", "giving", "tell", "told", "tells", "telling", 
    "work", "worked", "works", "working", "call", "called", "calls", "calling", 
    "try", "tried", "tries", "trying", "ask", "asked", "asks", "asking", 
    "need", "needed", "needs", "needing", "feel", "felt", "feels", "feeling", 
    "become", "became", "becomes", "becoming", "leave", "left", "leaves", "leaving", 
    "put", "puts", "putting", "mean", "meant", "means", "meaning", 
    "keep", "kept", "keeps", "keeping", "let", "lets", "letting", 
    "begin", "began", "begun", "begins", "beginning", "seem", "seemed", "seems", "seeming", 
    "help", "helped", "helps", "helping", "talk", "talked", "talks", "talking", 
    "turn", "turned", "turns", "turning", "start", "started", "starts", "starting", 
    "show", "showed", "shown", "shows", "showing", "hear", "heard", "hears", "hearing", 
    "play", "played", "plays", "playing", "run", "ran", "runs", "running", 
    "move", "moved", "moves", "moving", "like", "liked", "likes", "liking", 
    "live", "lived", "lives", "living", "believe", "believed", "believes", "believing", 
    "hold", "held", "holds", "holding", "bring", "brought", "brings", "bringing", 
    "happen", "happened", "happens", "happening", "write", "wrote", "written", "writes", "writing", 
    "provide", "provided", "provides", "providing", "sit", "sat", "sits", "sitting", 
    "stand", "stood", "stands", "standing", "lose", "lost", "loses", "losing", 
    "pay", "paid", "pays", "paying", "meet", "met", "meets", "meeting", 
    "include", "included", "includes", "including", "continue", "continued", "continues", "continuing", 
    "set", "sets", "setting", "learn", "learned", "learnt", "learns", "learning", 
    "change", "changed", "changes", "changing", "lead", "led", "leads", "leading", 
    "understand", "understood", "understands", "understanding", "watch", "watched", "watches", "watching", 
    "follow", "followed", "follows", "following", "stop", "stopped", "stops", "stopping", 
    "create", "created", "creates", "creating", "speak", "spoke", "spoken", "speaks", "speaking", 
    "read", "reads", "reading", "allow", "allowed", "allows", "allowing", 
    "add", "added", "adds", "adding", "spend", "spent", "spends", "spending", 
    "grow", "grew", "grown", "grows", "growing", "open", "opened", "opens", "opening", 
    "walk", "walked", "walks", "walking", "win", "won", "wins", "winning", 
    "offer", "offered", "offers", "offering", "remember", "remembered", "remembers", "remembering", 
    "love", "loved", "loves", "loving", "consider", "considered", "considers", "considering", 
    "appear", "appeared", "appears", "appearing", "buy", "bought", "buys", "buying", 
    "wait", "waited", "waits", "waiting", "serve", "served", "serves", "serving", 
    "die", "died", "dies", "dying", "send", "sent", "sends", "sending", 
    "expect", "expected", "expects", "expecting", "build", "built", "builds", "building", 
    "stay", "stayed", "stays", "staying", "fall", "fell", "fallen", "falls", "falling", 
    "cut", "cuts", "cutting", "reach", "reached", "reaches", "reaching", 
    "kill", "killed", "kills", "killing", "remain", "remained", "remains", "remaining",
    "do", "did", "done", "does", "doing", "have", "had", "has", "having", "be", "is", "am", "are", "was", "were", "been", "being"
]
custom_stop_words.extend(common_verbs)

# Dynamically add all years from 1880 to 1950 to the stop words list
year_stopwords = [str(year) for year in range(1880, 1951)]
custom_stop_words.extend(year_stopwords)

# Merge standard English stopwords with our custom list
ALL_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
print(f"Loaded {len(ALL_STOP_WORDS)} total stopwords to filter out noise (including years 1880-1950 and common verbs).")

# Load the JSON dataset
with open(GANDHI_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Dataset loaded successfully! Total original documents: {len(df):,}")

print("\n" + "=" * 70)
print("2. DATA PREPROCESSING & E5-OPTIMIZED CHUNKING")
print("=" * 70)


def clean_and_extract_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if isinstance(text, list):
        text = ' '.join([str(item) for item in text if item is not None])
    return str(text).strip()


df['clean_text'] = df['contents'].apply(clean_and_extract_text)

df_filtered = df[df['clean_text'] != ""].copy()
print(f"Documents with valid text retained: {len(df_filtered):,}")

# ---------------------------------------------------------
# CHUNKING LOGIC FOR 'e5-base-v2' (512 Token Limit)
# ---------------------------------------------------------
WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 10

chunked_data = []

for idx, row in df_filtered.iterrows():
    text = row['clean_text']
    words = text.split()

    if len(words) <= WORDS_PER_CHUNK:
        chunked_data.append({
            'doc_id': idx,
            'chunk_text': "passage: " + text
        })
    else:
        for i in range(0, len(words), WORDS_PER_CHUNK - OVERLAP_WORDS):
            chunk_words = words[i:i + WORDS_PER_CHUNK]
            if len(chunk_words) < 10:
                continue
            chunked_data.append({
                'doc_id': idx,
                'chunk_text': "passage: " + " ".join(chunk_words)
            })

df_chunks = pd.DataFrame(chunked_data)

print(f"Target Max Tokens: 450 (~{WORDS_PER_CHUNK} words)")
print(f"Target Overlap: 64 characters (~{OVERLAP_WORDS} words)")
print(f"Total chunks created: {len(df_chunks):,}")

# Extract list for BERTopic
docs = df_chunks['chunk_text'].tolist()


print("\n" + "=" * 70)
print("3. TRAINING BERTopic WITH INTFLOAT/E5-BASE-V2")
print("=" * 70)

# A. Embedding Model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading 'e5-base-v2' onto {device.upper()}...")
embedding_model = SentenceTransformer("intfloat/e5-base-v2", device=device)

# B. Dimensionality Reduction
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# C. Clustering Model
hdbscan_model = HDBSCAN(min_cluster_size=100, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# D. Vectorizer (USING OUR CUSTOM STOPWORDS)
vectorizer_model = CountVectorizer(stop_words=ALL_STOP_WORDS, ngram_range=(1, 2))

# Initialize BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    verbose=True
)

print(f"Encoding and fitting model on {len(docs):,} chunks...")
topics, probs = topic_model.fit_transform(docs)

print("Training complete!")

print("\n" + "=" * 70)
print("4. SAVING RESULTS")
print("=" * 70)

# Get the dataframe of topics (Topic -1 is the "noise" bin)
topic_info = topic_model.get_topic_info()
print(f"\nDiscovered {len(topic_info) - 1} distinct topics (excluding noise).")
print("\nTop 15 Largest Topics")
print(topic_info.head(16)[['Topic', 'Count', 'Name']])

# Save to CSV
topic_info.to_csv(os.path.join(RESULTS_DIR, 'bertopic_summary_e5.csv'), index=False)
print(f"Saved topic summary CSV to: {os.path.join(RESULTS_DIR, 'bertopic_summary_e5.csv')}")

# Save BERTopic model for later visualization (without the large embedding model)
MODEL_DIR = os.path.join(RESULTS_DIR, "bertopic_model")
topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
print(f"Saved BERTopic model to: {MODEL_DIR}")

print(f"\nComplete! Run GenerateViz.py to create interactive HTML visualizations.")
