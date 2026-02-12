import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Replace with your OMDb API key
API_KEY = "929728e2"

# ✅ Load dataset
df = pd.read_csv(r"C:\Users\rahul\OneDrive\Desktop\netflix_recommender\netflix_titles_cleaned.csv")

# Handle missing values
df['listed_in'] = df['listed_in'].fillna('')
df['description'] = df['description'].fillna('')
df['release_year'] = df['release_year'].fillna(0)
df['duration'] = df['duration'].fillna('0')

# --- Feature engineering for RandomForest ---
def convert_duration(value):
    if pd.isna(value):
        return 0
    if 'Season' in str(value):
        return int(value.split(' ')[0]) * 30  # approx 30 min per season
    if 'min' in str(value):
        return int(value.split(' ')[0])
    return 0

df['duration_numeric'] = df['duration'].apply(convert_duration)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['type'])  # Movie=0, TV Show=1
X = df[['release_year', 'duration_numeric']]

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X, y)

def predict_type(title):
    """Predict whether given title is Movie or TV Show"""
    row = df[df['title'] == title]
    if row.empty:
        return "Unknown"
    features = row[['release_year', 'duration_numeric']]
    pred = rf.predict(features)
    return le.inverse_transform(pred)[0]

# --- Content-Based Recommendation ---
df['combined_features'] = df['listed_in'] + ' ' + df['description']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def fetch_omdb_data(title):
    """Fetch poster, IMDb link, and plot"""
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
        data = requests.get(url).json()
        return {
            "poster": data.get("Poster") if data.get("Poster") and data["Poster"] != "N/A" else "/static/default_poster.jpg",
            "imdb_url": f"https://www.imdb.com/title/{data.get('imdbID')}/" if data.get("imdbID") else "#",
            "plot": data.get("Plot", "No description available.")
        }
    except:
        return {
            "poster": "/static/default_poster.jpg",
            "imdb_url": "#",
            "plot": "No description available."
        }

def get_content_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'listed_in', 'release_year', 'description']]

# Add a fake popularity metric
df['popularity_score'] = (df['release_year'] - df['release_year'].min()) / \
                         (df['release_year'].max() - df['release_year'].min())

def get_hybrid_recommendations(title, weight_content=0.7, weight_popularity=0.3):
    recs = get_content_recommendations(title)
    if len(recs) == 0:
        return []
    recs = recs.merge(df[['title', 'popularity_score']], on='title', how='left')
    recs['hybrid_score'] = weight_content * 1.0 + weight_popularity * recs['popularity_score']
    recs = recs.sort_values('hybrid_score', ascending=False).head(6)

    final_data = []
    for _, row in recs.iterrows():
        movie_data = fetch_omdb_data(row['title'])
        rec_type = predict_type(row['title'])
        final_data.append({
            'title': row['title'],
            'listed_in': row['listed_in'],
            'release_year': row['release_year'],
            'poster': movie_data['poster'],
            'imdb_url': movie_data['imdb_url'],
            'plot': movie_data['plot'],
            'type': rec_type
        })
    return final_data
