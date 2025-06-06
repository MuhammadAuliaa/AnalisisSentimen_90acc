from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response, jsonify
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
import io
import util
import time

matplotlib.use('Agg')  # Use non-GUI backend for rendering

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ✅ Define upload folder before using it
UPLOAD_FOLDER = "data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    df = pd.read_csv('data/dataSentimen.csv') 
    df = df[['Message', 'sentimen_vader']]
    data = df.to_dict(orient='records')
    return render_template('index.html', data=data, columns=df.columns)

@app.route("/generate_chart")
def generate_chart():
    df = pd.read_csv('data/dataSentimen.csv') 
    sentiment_counts = df['sentimen_vader'].value_counts().reindex(["Negatif", "Positif", "Netral"], fill_value=0)

    plt.figure(figsize=(6, 4))
    custom_palette = {'Negatif': '#e74a3b', 'Positif': '#0384fc', 'Netral': "#03fc9d"}
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=custom_palette)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.title("Distribusi Atribut Sentimen")
    plt.xlabel("Atribut Sentimen")
    plt.ylabel("Jumlah")

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    plt.close()
    img.seek(0)

    return send_file(img, mimetype="image/png")

def generate_wordcloud(text, colormap):
    if not text.strip():
        text = "No Data Available" 
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
    
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    
    return img.getvalue()

@app.route("/wordcloud_positive")
def wordcloud_mixed():
    """Generate WordCloud from all text in the dataset."""
    try:
        df = pd.read_csv('data/dataSentimen.csv')
        all_messages = ' '.join(df['Message'].dropna().astype(str))
        return Response(generate_wordcloud(all_messages, 'Blues'), mimetype='image/png')
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return Response(status=500)