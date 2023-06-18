import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, re, json, string, pickle
import nltk
# nltk.download('stopwords')
import seaborn as sns
import swifter

from cleantext import clean
from datetime import datetime
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

vect = TfidfVectorizer()

# SideBar
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Preprocessing','Naïve Bayes', 'Prediksi'], 
        icons=['house', 'gear', 'box','triangle'], menu_icon="cast", default_index=0)

if selected == 'Home':
    st.markdown("""
    <h1 style='text-align: center;'>Universitas Budi Luhur</h1>
    <h3 style='text-align: center;'>ANALISIS SENTIMEN TERHADAP ULASAN PENGGUNA APLIKASI WAHYOO MENGGUNAKAN METODE NAÏVE BAYES CLASSIFIER</h3>
    <h5 style='text-align: center;'>Analisis sentimen dapat menggunakan teknik text mining. Yang dapat mengklasifikasikan konten opini dari sumber data yang sangat banyak. Metode text mining untuk melakukan analisis sentimen terhadapat ulasan sebuah aplikasi di Google Play Store yaitu menggunakan algoritma Naïve Bayes karena proses analisis bersifat klasifikasi.</h5>
    """, unsafe_allow_html=True)

elif(selected == 'Preprocessing'):
    st.title('Preprocessing')
    uploaded_file = st.file_uploader('Pilih file')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.DataFrame(df)
        df = df.dropna()
        st.write(df)

        if st.button('Preprocessing', key='process'):
            # Case Folding
            def caseFolding(ulasan):
                ulasan = ulasan.lower()
                ulasan = ulasan.strip(' ')
                return ulasan
            df['content'] = df['content'].apply(caseFolding)

            # cleansing
            def cleansing(text):
                # remove newline characters
                t1 = text.replace('\\n', ' ')
                # remove emoji
                t2 =  clean(t1, no_emoji=True)
                # Replace URL (http:// atau https://)
                t3 = re.sub(r'(http|https):\/\/.*[\r\n]*', '', t2)
                # Replace #_something)_
                t4 = re.sub('#+', '', t3)
                # Replace word repetition with a single accurance ('ooooooooo' become 'oo') 
                t5 = re.sub(r'(.)\1+', r'\1\1', t4)
                # Replace punctuation repetition with a single accurance ('!!!!!!!!' becomes '!')
                t6 = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', t5)
                # Alphabets only, exlcude number and special charaters
                t7 = re.sub(r'[^a-zA-Z]', ' ', t6)
                return t7

            for i,r, in df.iterrows():
                y = cleansing(r['content'])
                df.loc[i, 'cleansing'] = y
            
            # remove duplicates
            df.drop_duplicates(subset=['cleansing'], keep='first', inplace=True)

            # Normalisasi kata
            kamus_normalisasi = pd.read_csv('/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/assets/kamus_slangwords.csv', sep=';')

            def text_normalize(text):
                text = ' '.join([kamus_normalisasi[kamus_normalisasi['kata_slang'] == word]['kata_benar'].values[0]
                if(kamus_normalisasi['kata_slang'] == word).any()
                else word for word in text.split()
                ])
                print('text_slang: ', text)
                text = str.lower(text)
                return text
            df['normalized'] = df['cleansing'].apply(lambda x: text_normalize(x))

            # Tokenization (memisahkan setiap kata)
            def tokenization(newtext):
                text = re.split('\W+', newtext)
                return text

            df['Tokenization'] = df['normalized'].apply(lambda x:tokenization(x.lower()))

            # Stopword removal (menghapus kata yang tidak penting)
            stopword = nltk.corpus.stopwords.words('/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/assets/indonesia-stopwords.txt')

            def remove_stopwords(newtext):
                newtext = [word for word in newtext if word not in stopword]
                return newtext

            df['Stopword_Removal'] = df['Tokenization'].apply(lambda x: remove_stopwords(x))

            # Stemmer (mengubah kata imbuhan menjadi kata dasar)
            def stemming(comments):
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                texts = []
                for w in comments:
                    dt = stemmer.stem(w)
                    texts.append(dt)
                text_results = []
                text_results = ' '.join(texts)
                return text_results
            
            df['Stemmed'] = df['Stopword_Removal'].apply(lambda x: stemming(x))

            st.write('Hasil Setelah Preprocessing')
            st.write(df)

            dictname = '/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/data/hasil_preprocessing.csv'
            df.to_csv(dictname)
            st.write('Data berhasil didownload!')
            
elif(selected == 'Naïve Bayes'):
    st.title('Naïve Bayes')
    uploaded_file = st.file_uploader('Pilih file')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = pd.DataFrame(df)
        df = df.dropna()

        df.head()
        x = df['Stemmed']
        y = df['Label']

        # MODELLING
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, stratify=y, shuffle=True,random_state=42)

        # Vectorizer
        vectorizer = CountVectorizer(binary=True)

        # tokenize and make the document into matrix
        vectorizer.fit(list(x_train) + list(x_test))
        pickle.dump(vectorizer, open("/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/models/feature_count_vect.sav", "wb"))

        # Transform document to document term matrix
        x_train_vec = vectorizer.transform(x_train)
        x_test_vec = vectorizer.transform(x_test)
        
        data_tabular_vectorizer = pd.DataFrame(x_test_vec.toarray(), columns=vectorizer.get_feature_names_out())
        st.write('Hasil CountVectorizer')
        st.write(data_tabular_vectorizer)

        # Membuat model prediksi
        clf = MultinomialNB()
        model = clf.fit(x_train_vec, y_train)
        prediction = clf.predict(x_test_vec)

        testing_results = []
        for i in range(len(prediction)):
            testing_results.append({'content': df['Stemmed'][i], 'score': df['score'][i],'Label': df['Label'][i],'class': prediction[i]})

        st.write('Hasil pengujian')
        data = pd.DataFrame(testing_results)
        st.write(data)

        # cari akurasi
        acc = accuracy_score(y_test, prediction)
        percentage = '{:.0%}'.format(acc)
        acc_percentage = f'{acc:.0%}'

        # Precision
        precision = precision_score(y_test, prediction, average='binary', pos_label='Negatif')
        percentage = '{:.0%}'.format(precision)
        prec_percentage = f'{precision:.0%}'

        # Recall
        recall = recall_score(y_test, prediction, average='binary', pos_label='Negatif')
        percentage = '{:.0%}'.format(recall)
        rec_percentage = f'{recall:.0%}'
        data_evaluasi = [['Akurasi', acc_percentage], ['Presisi', prec_percentage], ['Recall', rec_percentage]]
        st.markdown("""
        <style>
            table {
                'font-family': arial, sans-serif;
                'border-collapse': collapse;
                'width': 100%;
            }

            td, th {
                'border': 1px solid #dddddd;
                'text-align': left;
                'padding': 8px;
            }

            tr:nth-child(even) {
                'background-color': #dddddd;
            }
        </style>
        <table>
            <tr>
                <td>Akurasi</td>
                <td>{str(acc_percentage)}</td>
            </tr>
            <tr>
                <td>Presisi</td>
                <td>{str(prec_percentage)}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{str(rec_percentage)}</td>
            </tr>
        </table>
        """,unsafe_allow_html=True)
        evaluasi_hasil_df = pd.DataFrame(data_evaluasi, columns=['Nilai', 'Persentase'])

        st.table(evaluasi_hasil_df)

        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

        # confusion matrix
        cm = confusion_matrix(y_test, prediction,labels=['Positif', 'Negatif'])
        print(f'confusion matrix:\n {cm}')
        print('============================================================')
        print('classf-report',classification_report(y_test, prediction, zero_division=0))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Positif', 'Negatif'])

        disp.plot()
        fig = plt.show()
        st.write('Confusion Matrix')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

        # Plotting Pie
        def pct_pie(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return '{:.1f}%\n{:d}'.format(pct, absolute)

        sentimen = data['content'].groupby(data['class']).count().values
        print('sentimen: ', sentimen)

        plt.figure(figsize = (8,8))
        plt.pie(sentimen, explode=(0,0), labels=['Negatif', 'Positif'], shadow = False, 
                autopct=lambda pct: pct_pie(pct, sentimen), startangle=90)
        plt.title('Perbandingan Sentimen', fontsize=18)
        plt.axis('equal')
        plt.legend(fontsize=11)
        fig = plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

        dictname = '/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/models/trained_model.sav'

        if st.button('Download Model'):
            with open(dictname, "wb") as f:
                pickle.dump(model, f)
            st.write('Model berhasil didownload!')

elif (selected == 'Prediksi'):
    dictname = '/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/models/trained_model.sav'
    count_vect_path = '/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/models/feature_count_vect.sav'

    # try:
    model_fraud = pickle.load(open(dictname, 'rb'))
    print('model_fraud: ', model_fraud)
    # finally:
    #     model_fraud.close()
    # with open('outputs/models/trained_model.sav', 'rb') as f:
    #     model_fraud = pickle.load(f)

    vectorizer = pickle.load(open(count_vect_path, "rb"))

    # loaded_vec = TfidfVectorizer(decode_error='replace', vocabulary=set(pickle.load(open('/Users/nezuko/Documents/tugas_akhir/web-app-ta/ilham-svm-ta/outputs/models/feature_tf-idf.sav', 'rb'))))

    # vect = TfidfVectorizer()

    # Judul Halaman
    st.title('Prediksi Ulasan Aplikasi Wahyoo')

    clean_teks = st.text_input('Masukkan ulasan anda')
    print('c-txt: ', clean_teks)

    fraud_detection = ''

    if (clean_teks == ''):
        if(st.button('Hasil Deteksi', disabled=True)):    
            predict_fraud = model_fraud.predict(vectorizer.transform([clean_teks]).toarray())

            if(predict_fraud[0] == 'Positif'):
                fraud_detection = 'Sentimen Positif'
            elif(predict_fraud[0] == 'Negatif'):
                fraud_detection = 'Sentimen Negatif'
            else:
                fraud_detection = 'Tidak Terdeteksi'
    elif (clean_teks != ''):
        if(st.button('Hasil Deteksi')):
            predict_fraud = model_fraud.predict(vectorizer.transform([clean_teks]).toarray())
            print('predict_fraud: ', predict_fraud)

            if(predict_fraud[0] == 'Positif'):
                fraud_detection = 'Sentimen Positif'
            elif(predict_fraud[0] == 'Negatif'):
                fraud_detection = 'Sentimen Negatif'
            else:
                fraud_detection = 'Tidak Terdeteksi'

    st.success(fraud_detection)        
