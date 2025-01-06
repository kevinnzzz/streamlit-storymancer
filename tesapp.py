import streamlit as st
import spacy
import pickle
import os
import sys
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS

# 🛠️ Mengatur Path File
if getattr(sys, 'frozen', False):
    current_dir = sys._MEIPASS2
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))

model_alur_path = os.path.join(current_dir, 'model_alur_naive_bayes.pkl')
model_tema_path = os.path.join(current_dir, 'model_tema_naive_bayes.pkl')
vectorizer_path = os.path.join(current_dir, 'model_vectorizer.pkl')
csv_orang_path = os.path.join(current_dir, 'orang.csv')
csv_tempat_path = os.path.join(current_dir, 'datatempat.csv')

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

model_alur = load_pickle(model_alur_path)
model_tema = load_pickle(model_tema_path)
vectorizer = load_pickle(vectorizer_path)

# 🔍 Fungsi Analisis Cerita
def tokoh(text):
    orang = set()
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    for char in punctuation:
        text = text.replace(char, "")
    with open(csv_orang_path, 'r', encoding='utf-8') as file_orang:
        data_csv = [line.strip() for line in file_orang]
    for item in text.split():
        if item in data_csv:
            orang.add(item)
    return list(orang)

def latar_tempat(text):
    tempat = set()
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    for char in punctuation:
        text = text.replace(char, "")
    with open(csv_tempat_path, 'r', encoding='utf-8') as file_tempat:
        data_csv = [line.strip() for line in file_tempat]
    for item in text.split():
        if item in data_csv:
            tempat.add(item)
    return list(tempat)

def tema(text):
    transformed_text = vectorizer.transform([text])
    hasil_prediksi = model_tema.predict(transformed_text)
    label_dict = {1: 'Romantis', 2: 'Persahabatan', 3: 'Petualangan', 4: 'Perjuangan', 5: 'Religi'}
    return label_dict.get(hasil_prediksi[0], 'Label tidak dikenali')

def alur(text):
    transformed_text = vectorizer.transform([text])
    hasil_prediksi = model_alur.predict(transformed_text)
    label_dict = {1: 'Maju', 2: 'Mundur', 3: 'Campuran'}
    return label_dict.get(hasil_prediksi[0], 'Label tidak dikenali')

def ringkasancerita(text):
    stopwords = STOP_WORDS
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'\t\n\r\x0b\x0c'"
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1
    max_frequency = max(word_frequencies.values(), default=1)
    word_frequencies = {word: freq / max_frequency for word, freq in word_frequencies.items()}
    sentence_tokens = list(doc.sents)
    sentence_scores = {sent: sum(word_frequencies.get(word.text.lower(), 0) for word in sent) for sent in sentence_tokens}
    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    return ' '.join(sent.text for sent in summary)

# 🎨 Tampilan Streamlit
st.title("📚 Story Mancer: Analisis Unsur Cerita")
st.write("Masukkan cerita Anda di bawah ini untuk dianalisis:")

# 📝 Input Teks Pengguna
story_text = st.text_area("📝 Tulis Cerita Anda di sini", height=300)

if st.button("🚀 Analisis Cerita"):
    if not story_text.strip():
        st.warning("⚠️ Silakan masukkan cerita terlebih dahulu.")
    else:
        tokoh_cerita = tokoh(story_text)
        latar_tempat_cerita = latar_tempat(story_text)
        tema_cerita = tema(story_text)
        alur_cerita = alur(story_text)
        ringkasan_cerita = ringkasancerita(story_text)

        st.subheader("✅ **Hasil Analisis:**")
        
        st.write("### 🧍 Tokoh:")
        if tokoh_cerita:
            for char in tokoh_cerita:
                st.write(f"- {char}")
        else:
            st.write("_(Tidak ditemukan tokoh yang jelas)_")
        
        st.write("### 🏞️ Latar Tempat:")
        if latar_tempat_cerita:
            for place in latar_tempat_cerita:
                st.write(f"- {place}")
        else:
            st.write("_(Tidak ditemukan latar tempat yang jelas)_")
        
        st.write("### 📖 Alur Cerita:")
        st.write(f"- **{alur_cerita}**")
        
        st.write("### 🎭 Tema Cerita:")
        st.write(f"- **{tema_cerita}**")
        
        st.write("### 📝 Ringkasan Cerita:")
        st.write(ringkasan_cerita)
