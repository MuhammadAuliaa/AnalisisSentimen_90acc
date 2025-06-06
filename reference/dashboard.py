# Training Data -> Evaluation
# hover siderbar (background abu muda -> hover merah)
# biru -> merah maroon
# eval page -> table -> feedback design
# analisis sentimen title ->
# bar 904

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

from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

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
    df = df[['full_text', 'tweet_english', 'Analysis']]
    data = df.to_dict(orient='records')
    return render_template('index.html', data=data, columns=df.columns)

@app.route("/generate_chart")
def generate_chart():
    df = pd.read_csv('data/dataSentimen.csv') 
    sentiment_counts = df['Analysis'].value_counts().reindex(["Negatif", "Positif"], fill_value=0)

    plt.figure(figsize=(6, 4))
    custom_palette = {'Negatif': '#e74a3b', 'Positif': '#0384fc'}
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
        all_messages = ' '.join(df['full_text'].dropna().astype(str))
        return Response(generate_wordcloud(all_messages, 'Blues'), mimetype='image/png')
    except Exception as e:
        print(f"Error generating wordcloud: {e}")
        return Response(status=500)

# @app.route('/preprocessing')
# def preprocessing():
#     df = pd.read_csv("data/dataCleaning.csv")
#     df = df[['full_text']]
#     columns = df.columns.tolist()
#     data = df.to_dict(orient="records")
#     return render_template("preprocessing.html", columns=columns, data=data)

# Load CSV Data
def load_data_processing():
    df = pd.read_csv("data/dataCleaning.csv")
    df = df.sample(50)
    df = df[['full_text']]
    return df

# Preprocessing Function
def preprocess_data(df):
    df['full_text'] = df['full_text'].astype(str).str.lower()
    df['full_text'] = df['full_text'].replace('false', pd.NA)
    df = df.dropna(subset=['full_text'])

    df['full_text'] = df['full_text'].astype(str).str.lower()
    df = df[df['full_text'] != 'false']
    df['full_text'] = df['full_text'].apply(util.clean)

    min_words = 2
    max_words = 50
    df = util.filter_tokens_by_length(df, 'full_text', min_words, max_words)

    df['full_text'] = df['full_text'].apply(util.normalisasi)
    df['full_text'] = df['full_text'].apply(util.stopword)
    df['full_text'] = df['full_text'].apply(util.tokenisasi)
    df['full_text'] = df['full_text'].apply(util.stemming)

    return df[['full_text']]

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        # Simulate Processing Delay
        time.sleep(2)

        # Load Data
        df = pd.read_csv("data/dataCleaning.csv")
        df = df.head(50)
        df = df[['full_text']]

        # Preprocessing Steps
        df['full_text'] = df['full_text'].astype(str).str.lower()
        df['full_text'] = df['full_text'].replace('false', pd.NA)
        df = df.dropna(subset=['full_text'])
        df['full_text'] = df['full_text'].apply(util.clean)
        df = util.filter_tokens_by_length(df, 'full_text', 2, 50)
        df['full_text'] = df['full_text'].apply(util.normalisasi)
        df['full_text'] = df['full_text'].apply(util.stopword)
        df['full_text'] = df['full_text'].apply(util.tokenisasi)
        df['full_text'] = df['full_text'].apply(util.stemming)

        processed_data = df['full_text'].tolist()

        return jsonify({"processed_data": processed_data})

    # Initial Data (Before Preprocessing)
    df = pd.read_csv("data/dataCleaning.csv")
    df = df[['full_text']]
    df = df.head(50)
    columns = df.columns.tolist()
    data = df.to_dict(orient="records")

    return render_template("preprocessing.html", columns=columns, data=data)

# @app.route("/translate")
# def translate():
#     df = pd.read_csv('data/dataPreprocessing.csv')  # Read the CSV file
#     data = df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
#     columns = df.columns.tolist()  # Get column names
#     return render_template("translate.html", data=data, columns=columns)

@app.route("/translate", methods=["GET", "POST"])
def translate():
    df = pd.read_csv('data/dataPreprocessing.csv')  # Load dataset
    df = df.head(10)
    data = df.to_dict(orient="records")  # Convert to dictionary
    columns = df.columns.tolist()  # Extract column names

    translated_data = None
    translation_status = None

    if request.method == "POST" and "translate" in request.form:
        if "full_text" in df.columns:
            df = df.dropna(subset=["full_text"])  # Remove NaN rows
            translated_data = df["full_text"].apply(util.translate_tweet).tolist()
            df["translated_text"] = translated_data  # Add new column
            
            # Save the translated data to a new CSV file
            # df.to_csv('data/dataTranslated.csv', index=False)
            
            translation_status = f"Translated {len(translated_data)} rows successfully!"
        else:
            translation_status = "Column 'full_text' not found in the dataset."

    return render_template("translate.html", data=data, columns=columns, translated_data=translated_data, translation_status=translation_status)

@app.route("/labeling", methods=["GET", "POST"])
def labeling():
    df = pd.read_csv('data/dataTerjemahan.csv')  # Read the CSV file
    df = df[['full_text', 'tweet_english']]

    if request.method == "POST":
        if 'tweet_english' in df.columns:
            df['Subjectivity'] = df['tweet_english'].apply(util.getSubjectivity)
            df['Polarity'] = df['tweet_english'].apply(util.getPolarity)
            df['Analysis'] = df['Polarity'].apply(util.analyze)
            message = f"Labeling {len(df)} rows Successful!"
        else:
            message = "Data yang dimasukkan tidak sesuai."
    else:
        message = None

    data = df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    columns = df.columns.tolist()  # Get column names

    return render_template("labeling.html", data=data, columns=columns, message=message)

# @app.route("/indobert")
# def indobert():
#     df = pd.read_csv('data/dataSentimen.csv')  # Read the CSV file
#     df = df[['full_text', 'tweet_english', 'Analysis']]
#     data = df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
#     columns = df.columns.tolist()  # Get column names
#     return render_template("indobert.html", data=data, columns=columns)

@app.route("/indobert", methods=["GET", "POST"])
def indobert():
    df = pd.read_csv('data/dataSentimen.csv')  
    df = df[['full_text', 'tweet_english', 'Analysis']]

    if request.method == "POST":
        try:
            # Ambil nilai input dari form
            epochs = int(request.form.get("epochs", 2))
            test_size = float(request.form.get("test_size", 0.2))

            # Hapus duplikasi
            df = df.drop_duplicates(subset=['full_text'])

            # Load model IndoBERT
            model_name = 'indobenchmark/indobert-base-p1'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

            # Tokenisasi dan label encoding
            reviews = df['full_text'].tolist()
            labels = df['Analysis'].tolist()

            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)

            input_ids = []
            attention_masks = []
            max_length = 128  

            for review in reviews:
                encoded_dict = tokenizer.encode_plus(
                    review,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='np'
                )
                input_ids.append(encoded_dict['input_ids'].tolist())  # Perbaikan dari .numpy()
                attention_masks.append(encoded_dict['attention_mask'].tolist())  # Perbaikan dari .numpy()

            input_ids = np.array(input_ids).squeeze(axis=1)
            attention_masks = np.array(attention_masks).squeeze(axis=1)
            labels = np.array(labels)

            # Split data
            train_input_ids, test_input_ids, train_labels, test_labels = train_test_split(
                input_ids, labels, test_size=test_size, random_state=42
            )

            train_attention_masks, test_attention_masks = train_test_split(
                attention_masks, test_size=test_size, random_state=42
            )

            # Konversi ke TensorFlow tensor
            train_input_ids = tf.convert_to_tensor(train_input_ids, dtype=tf.int32)
            test_input_ids = tf.convert_to_tensor(test_input_ids, dtype=tf.int32)
            train_attention_masks = tf.convert_to_tensor(train_attention_masks, dtype=tf.int32)
            test_attention_masks = tf.convert_to_tensor(test_attention_masks, dtype=tf.int32)
            train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
            test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)

            # Training model
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # FIXED
            metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

            history = model.fit(
                [train_input_ids, train_attention_masks],
                train_labels,
                batch_size=8,
                epochs=epochs,
                validation_data=([test_input_ids, test_attention_masks], test_labels)
            )

            # Evaluasi model
            test_predictions = model.predict([test_input_ids, test_attention_masks])
            predicted_labels = tf.argmax(test_predictions.logits, axis=1)
            report = classification_report(test_labels.numpy(), predicted_labels.numpy(), output_dict=True)

            return render_template("indobert.html", data=df.to_dict(orient="records"), 
                                   columns=df.columns.tolist(), message="Training Completed!", 
                                   report=report)

        except Exception as e:
            return render_template("indobert.html", data=df.to_dict(orient="records"), 
                                   columns=df.columns.tolist(), message=f"Error: {str(e)}")

    return render_template("indobert.html", data=df.to_dict(orient="records"), columns=df.columns.tolist())


if __name__ == "__main__":
    # Ensure static folder exists for storing images
    if not os.path.exists("static"):
        os.makedirs("static")

    app.run(debug=True)
