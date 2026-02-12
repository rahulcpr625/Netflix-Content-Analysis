from flask import Flask, render_template, request
from recommender import get_hybrid_recommendations, df, predict_type

app = Flask(__name__)

@app.route('/')
def home():
    titles = sorted(df['title'].dropna().unique()[:200])
    return render_template('index.html', titles=titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations = get_hybrid_recommendations(title)
    predicted_type = predict_type(title)
    return render_template('results.html', title=title, recommendations=recommendations, predicted_type=predicted_type)

if __name__ == '__main__':
    app.run(debug=True)
