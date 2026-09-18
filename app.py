import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import io

# NLP & Stats
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import chi2_contingency
import prince

# Page Configuration
st.set_page_config(
    page_title="IndoTextMiner - KH Coder Alternative",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. PREPROCESSING & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_nlp_tools():
    """Load Sastrawi Indonesian Stopwords and Stemmer."""
    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())
    
    # Add custom common Indonesian stopwords/fillers if needed
    custom_stopwords = {'ya', 'tidak', 'ada', 'dan', 'yang', 'di', 'ke', 'dari', 'ini', 'itu'}
    stopwords.update(custom_stopwords)
    
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    return stopwords, stemmer

stopwords_id, stemmer_id = load_nlp_tools()

def preprocess_text(text, do_stemming=False):
    """Clean and tokenize Bahasa Indonesia text."""
    if not isinstance(text, str):
        return []
    # Lowercase & remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    
    # Filter stopwords & short tokens
    tokens = [t for t in tokens if t not in stopwords_id and len(t) > 2]
    
    if do_stemming:
        tokens = [stemmer_id.stem(t) for t in tokens]
        
    return tokens

def get_kwic(text_series, keyword, window=5):
    """Keyword in Context (KWIC) search."""
    kwic_results = []
    pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
    
    for idx, text in enumerate(text_series):
        if not isinstance(text, str):
            continue
        words = text.split()
        for i, word in enumerate(words):
            if pattern.search(word):
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                left_context = " ".join(words[start:i])
                matched = words[i]
                right_context = " ".join(words[i+1:end])
                kwic_results.append({
                    "Doc ID": idx + 1,
                    "Konteks Kiri (Left)": left_context,
                    "Kata Kunci (Keyword)": matched,
                    "Konteks Kanan (Right)": right_context
                })
    return pd.DataFrame(kwic_results)

# -----------------------------------------------------------------------------
# 2. UI SIDEBAR - DATA UPLOAD & CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.title("🇮🇩 IndoTextMiner")
st.sidebar.markdown("**Aplikasi Analisis Teks & Linguistik Komputasi**")

# Updated file uploader to accept multiple formats
uploaded_file = st.sidebar.file_uploader(
    "1. Unggah Berkas Dokumen", 
    type=["csv", "xls", "xlsx", "txt", "md"]
)

if uploaded_file is not None:
    # -------------------------------------------------------------------------
    # FILE PARSING LOGIC
    # -------------------------------------------------------------------------
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_ext in ['txt', 'md']:
            # Read plain text and split by new lines to mimic document rows
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            lines = [line.strip() for line in stringio.readlines() if line.strip()]
            df = pd.DataFrame(lines, columns=["text"])
            st.sidebar.info("Berkas teks/markdown dimuat. Setiap baris/paragraf diubah menjadi satu entri data.")
            
        st.sidebar.success(f"Berkas berhasil dimuat: {df.shape[0]} baris.")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat berkas: {e}")
        st.stop()
    
    text_column = st.sidebar.selectbox("2. Pilih Kolom Teks Utama", df.columns)
    
    # Categorical column for characteristic words / crosstab
    cat_columns = [None] + list(df.columns)
    category_column = st.sidebar.selectbox("3. Pilih Kolom Kategori/Bagian (Opsional)", cat_columns)
    
    do_stemming = st.sidebar.checkbox("Gunakan Stemming Sastrawi (Lebih lambat)", value=False)
    
    # Preprocess Data
    with st.spinner("Memproses teks Bahasa Indonesia..."):
        df['tokens'] = df[text_column].apply(lambda x: preprocess_text(x, do_stemming=do_stemming))
        df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Frekuensi Kata", 
        "🔍 Konteks Kata (KWIC)", 
        "🌐 Jaringan Ko-okurensi", 
        "📈 Eksplorasi Ko-okurensi", 
        "🗺️ Analisis Korespondensi", 
        "🏷️ Kata Khas per Bagian"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: WORD FREQUENCY LIST
    # -------------------------------------------------------------------------
    with tab1:
        st.header("📊 Daftar Frekuensi Kata (Word Frequency List)")
        st.write("Menampilkan kata yang paling sering muncul dalam dokumen.")
        
        top_n = st.slider("Jumlah kata teratas:", 10, 100, 20, key="freq_slider")
        all_tokens = [token for tokens in df['tokens'] for token in tokens]
        freq_dist = Counter(all_tokens)
        freq_df = pd.DataFrame(freq_dist.most_common(top_n), columns=["Kata", "Frekuensi"])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(freq_df, use_container_width=True)
        with col2:
            fig = px.bar(freq_df, x="Frekuensi", y="Kata", orientation='h', 
                         title=f"Top {top_n} Kata Sering Muncul", color="Frekuensi",
                         color_continuous_scale="Viridis")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: CONTEXT WHERE A WORD IS USED (KWIC)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("🔍 Konteks Penggunaan Kata (KWIC - Keyword in Context)")
        st.write("Melihat kalimat atau paragraf di mana kata tertentu digunakan.")
        
        search_kw = st.text_input("Masukkan Kata Kunci yang Ingin Dicari:", "")
        window_size = st.slider("Ukuran Jendela (Kata Kiri/Kanan):", 2, 10, 5)
        
        if search_kw:
            kwic_df = get_kwic(df[text_column], search_kw, window=window_size)
            if not kwic_df.empty:
                st.write(f"Ditemukan **{len(kwic_df)}** Kemunculan:")
                st.dataframe(kwic_df, use_container_width=True)
            else:
                st.warning(f"Kata '{search_kw}' tidak ditemukan dalam dokumen.")

    # -------------------------------------------------------------------------
    # TAB 3: CO-OCCURRENCE NETWORK
    # -------------------------------------------------------------------------
    with tab3:
        st.header("🌐 Jaringan Ko-okurensi Kata (Co-occurrence Network)")
        st.write("Visualisasi interaktif hubungan antar kata yang sering muncul bersamaan.")
        
        col_net1, col_net2 = st.columns(2)
        with col_net1:
            max_words_net = st.slider("Jumlah Kata Teratas dalam Jaringan:", 10, 50, 25)
        with col_net2:
            min_cooc = st.slider("Ambang Batas Kemunculan Bersama (Minimum Co-occurrence):", 1, 10, 2)
            
        cv = CountVectorizer(max_features=max_words_net)
        X = cv.fit_transform(df['clean_text'])
        words = cv.get_feature_names_out()
        
        # Calculate Co-occurrence Matrix
        cooc_matrix = (X.T * X)
        cooc_matrix.setdiag(0) # Remove self-loop
        cooc_df = pd.DataFrame(cooc_matrix.toarray(), index=words, columns=words)
        
        # Build NetworkX Graph
        G = nx.Graph()
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                weight = cooc_df.iloc[i, j]
                if weight >= min_cooc:
                    G.add_edge(words[i], words[j], weight=int(weight))
                    
        if G.number_of_edges() > 0:
            net = Network(height="550px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)
            
            # Physics visual settings
            for node in net.nodes:
                node["size"] = G.degree(node["id"]) * 4 + 10
            
            net.save_graph("network.html")
            with open("network.html", 'r', encoding='utf-8') as f:
                html_data = f.read()
            components.html(html_data, height=570)
        else:
            st.warning("Tidak ada hubungan kata yang memenuhi ambang batas minimum.")

    # -------------------------------------------------------------------------
    # TAB 4: EXPLORING CO-OCCURRENCES (MDS & CLUSTER ANALYSIS)
    # -------------------------------------------------------------------------
    with tab4:
        st.header("📈 Metode Eksplorasi Ko-okurensi")
        st.write("Mengelompokkan kata berdasarkan kemiripan pola kemunculannya.")
        
        n_terms = st.slider("Jumlah Kata yang Diikutsertakan:", 10, 40, 20, key="exp_slider")
        cv_exp = CountVectorizer(max_features=n_terms)
        X_exp = cv_exp.fit_transform(df['clean_text']).toarray()
        words_exp = cv_exp.get_feature_names_out()
        
        # Jaccard / Cosine Distance matrix between words
        word_matrix = X_exp.T
        from sklearn.metrics.pairwise import cosine_distances
        dist_matrix = cosine_distances(word_matrix)
        
        exp_subtab1, exp_subtab2 = st.tabs(["Multidimensional Scaling (MDS)", "Hierarchical Clustering"])
        
        with exp_subtab1:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            coords = mds.fit_transform(dist_matrix)
            mds_df = pd.DataFrame(coords, columns=["Dimensi 1", "Dimensi 2"], index=words_exp)
            mds_df['Kata'] = words_exp
            
            fig_mds = px.scatter(mds_df, x="Dimensi 1", y="Dimensi 2", text="Kata", 
                                 title="Peta Multidimensional Scaling (MDS) Kata")
            fig_mds.update_traces(textposition='top center', marker=dict(size=12, color='DarkCyan'))
            st.plotly_chart(fig_mds, use_container_width=True)
            
        with exp_subtab2:
            st.subheader("Dendrogram Pengelompokan Kata")
            Z = linkage(dist_matrix, method='ward')
            import matplotlib.pyplot as plt
            fig_dend, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z, labels=words_exp, leaf_rotation=90, ax=ax)
            plt.ylabel("Jarak (Distance)")
            st.pyplot(fig_dend)

    # -------------------------------------------------------------------------
    # TAB 5: CORRESPONDENCE ANALYSIS OF WORDS
    # -------------------------------------------------------------------------
    with tab5:
        st.header("🗺️ Analisis Korespondensi Kata (Correspondence Analysis)")
        st.write("Memetakan hubungan antara kata dan dokumen / variabel kategori dalam ruang 2D.")
        
        if category_column and category_column != None:
            cv_ca = CountVectorizer(max_features=25)
            X_ca = cv_ca.fit_transform(df['clean_text']).toarray()
            words_ca = cv_ca.get_feature_names_out()
            
            crosstab_df = pd.DataFrame(X_ca, columns=words_ca)
            crosstab_df['Category'] = df[category_column].astype(str).values
            grouped_ct = crosstab_df.groupby('Category').sum()
            
            ca = prince.CA(n_components=2, random_state=42)
            ca = ca.fit(grouped_ct)
            
            col_coords = ca.column_coordinates(grouped_ct) # Words
            row_coords = ca.row_coordinates(grouped_ct)    # Categories
            
            ca_plot_df = pd.DataFrame({
                'Dim 1': list(col_coords[0]) + list(row_coords[0]),
                'Dim 2': list(col_coords[1]) + list(row_coords[1]),
                'Label': list(col_coords.index) + list(row_coords.index),
                'Tipe': ['Kata'] * len(col_coords) + ['Kategori'] * len(row_coords)
            })
            
            fig_ca = px.scatter(ca_plot_df, x="Dim 1", y="Dim 2", color="Tipe", text="Label",
                                title="Peta Analisis Korespondensi (Kata vs Kategori)")
            fig_ca.update_traces(textposition='top center', marker=dict(size=10))
            st.plotly_chart(fig_ca, use_container_width=True)
        else:
            st.info("Pilih 'Kolom Kategori/Bagian' di sidebar sebelah kiri untuk menjalankan Analisis Korespondensi.")

    # -------------------------------------------------------------------------
    # TAB 6: CHARACTERISTIC WORDS OF EACH PART (CROSSTAB / KEYNESS)
    # -------------------------------------------------------------------------
    with tab6:
        st.header("🏷️ Kata Khas per Bagian (Characteristic Words / Crosstab)")
        st.write("Menemukan kata-kata unik atau signifikan yang menjadi ciri khas kelompok/kategori tertentu.")
        
        if category_column and category_column != None:
            cv_char = CountVectorizer(max_features=50)
            X_char = cv_char.fit_transform(df['clean_text']).toarray()
            words_char = cv_char.get_feature_names_out()
            
            ct = pd.DataFrame(X_char, columns=words_char)
            ct['Group'] = df[category_column].astype(str).values
            ct_sum = ct.groupby('Group').sum()
            
            # Chi-square test for keyness
            chi2_stats = {}
            for word in words_char:
                obs = ct_sum[word].values
                total_words_per_group = ct_sum.sum(axis=1).values - obs
                contingency_table = np.array([obs, total_words_per_group])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi2_stats[word] = chi2
                
            keyness_df = pd.DataFrame(list(chi2_stats.items()), columns=['Kata', 'Skor Chi-Square'])
            keyness_df = keyness_df.sort_values(by='Skor Chi-Square', ascending=False)
            
            st.subheader("Kata Paling Signifikan secara Statistik (Chi-Square Keyness)")
            st.dataframe(keyness_df.head(15), use_container_width=True)
            
            st.subheader("Tabel Silang Frekuensi Kata per Kategori")
            st.dataframe(ct_sum.T, use_container_width=True)
        else:
            st.info("Pilih 'Kolom Kategori/Bagian' di sidebar sebelah kiri untuk melihat Kata Khas per Bagian.")

else:
    st.info("👋 Silakan unggah file dokumen (CSV, XLS, XLSX, TXT, MD) di sidebar kiri untuk memulai analisis.")