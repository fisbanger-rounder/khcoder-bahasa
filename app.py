import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import io
import logging

# NLP & Stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency
import prince

from textminer import get_kwic, preprocess_text

# Page Configuration
st.set_page_config(
    page_title="IndoTextMiner - KH Coder Alternative",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. PREPROCESSING & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
MAX_ROWS = 20_000


@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes, filename):
    """Parse an uploaded file into a DataFrame. Cached on file content."""
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'csv':
        return pd.read_csv(io.BytesIO(file_bytes))
    if ext in ('xls', 'xlsx'):
        return pd.read_excel(io.BytesIO(file_bytes))
    # Plain text / markdown: split on blank lines so paragraph boundaries survive.
    text = file_bytes.decode('utf-8', errors='replace')
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    df = pd.DataFrame({"text": paragraphs})
    df["Bagian"] = [f"Paragraf {i + 1}" for i in range(len(paragraphs))]
    return df


@st.cache_data(show_spinner=False)
def build_corpus(df, text_column, do_stemming):
    """Tokenize the corpus once; cached so slider moves don't re-stem everything."""
    tokens = df[text_column].apply(lambda x: preprocess_text(x, do_stemming=do_stemming))
    return tokens, tokens.apply(" ".join)

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
    try:
        df = load_dataframe(uploaded_file.getvalue(), uploaded_file.name)
    except Exception:
        logging.exception("Gagal memuat berkas unggahan")
        st.sidebar.error("Gagal memuat berkas: format tidak dikenali atau berkas rusak.")
        st.stop()

    if len(df) == 0:
        st.sidebar.error("Berkas tidak berisi baris data.")
        st.stop()
    if len(df) > MAX_ROWS:
        st.sidebar.error(f"Berkas terlalu besar: {len(df)} baris (batas {MAX_ROWS}).")
        st.stop()

    st.sidebar.success(f"Berkas berhasil dimuat: {df.shape[0]} baris.")
    if uploaded_file.name.rsplit('.', 1)[-1].lower() in ('txt', 'md'):
        st.sidebar.info("Berkas teks dimuat: setiap paragraf menjadi satu entri data (kolom `Bagian`).")

    text_column = st.sidebar.selectbox("2. Pilih Kolom Teks Utama", df.columns)

    # Categorical column for characteristic words / crosstab (the text column itself is meaningless here)
    category_column = st.sidebar.selectbox(
        "3. Pilih Kolom Kategori/Bagian (Opsional)",
        [None] + [c for c in df.columns if c != text_column],
    )

    do_stemming = st.sidebar.checkbox("Gunakan Stemming Sastrawi (Lebih lambat)", value=False)

    # Preprocess Data (underscore-prefixed so user columns are never overwritten)
    with st.spinner("Memproses teks Bahasa Indonesia..."):
        df['_tokens'], df['_clean_text'] = build_corpus(df, text_column, do_stemming)
    
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
        all_tokens = [token for tokens in df['_tokens'] for token in tokens]
        freq_dist = Counter(all_tokens)
        freq_df = pd.DataFrame(freq_dist.most_common(top_n), columns=["Kata", "Frekuensi"])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(freq_df)
        with col2:
            fig = px.bar(freq_df, x="Frekuensi", y="Kata", orientation='h', 
                         title=f"Top {top_n} Kata Sering Muncul", color="Frekuensi",
                         color_continuous_scale="Viridis")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig)

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
                st.dataframe(kwic_df)
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
        X = cv.fit_transform(df['_clean_text'])
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
            net = Network(height="550px", width="100%", bgcolor="#222222", font_color="white",
                          cdn_resources="in_line")
            net.from_nx(G)
            
            # Physics visual settings
            for node in net.nodes:
                node["size"] = G.degree(node["id"]) * 4 + 10
            
            # generate_html() returns the page; never write it to disk (shared CWD = cross-session leak)
            components.html(net.generate_html(), height=570)
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
        X_exp = cv_exp.fit_transform(df['_clean_text']).toarray()
        words_exp = cv_exp.get_feature_names_out()
        
        # Cosine distance matrix between words
        dist_matrix = cosine_distances(X_exp.T)
        
        exp_subtab1, exp_subtab2 = st.tabs(["Multidimensional Scaling (MDS)", "Hierarchical Clustering"])
        
        with exp_subtab1:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            coords = mds.fit_transform(dist_matrix)
            mds_df = pd.DataFrame(coords, columns=["Dimensi 1", "Dimensi 2"], index=words_exp)
            mds_df['Kata'] = words_exp
            
            fig_mds = px.scatter(mds_df, x="Dimensi 1", y="Dimensi 2", text="Kata", 
                                 title="Peta Multidimensional Scaling (MDS) Kata")
            fig_mds.update_traces(textposition='top center', marker=dict(size=12, color='DarkCyan'))
            st.plotly_chart(fig_mds)
            
        with exp_subtab2:
            st.subheader("Dendrogram Pengelompokan Kata")
            # linkage() on a square matrix silently recomputes pdist(y) with Euclidean metric,
            # discarding the cosine distances. Pass the condensed form instead.
            Z = linkage(squareform(dist_matrix, checks=False), method='average')
            import matplotlib.pyplot as plt
            fig_dend, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z, labels=words_exp, leaf_rotation=90, ax=ax)
            ax.set_ylabel("Jarak (Distance)")
            st.pyplot(fig_dend)
            plt.close(fig_dend)

    # -------------------------------------------------------------------------
    # TAB 5: CORRESPONDENCE ANALYSIS OF WORDS
    # -------------------------------------------------------------------------
    with tab5:
        st.header("🗺️ Analisis Korespondensi Kata (Correspondence Analysis)")
        st.write("Memetakan hubungan antara kata dan dokumen / variabel kategori dalam ruang 2D.")
        
        if category_column and category_column != None:
            cv_ca = CountVectorizer(max_features=25)
            X_ca = cv_ca.fit_transform(df['_clean_text']).toarray()
            words_ca = cv_ca.get_feature_names_out()
            
            crosstab_df = pd.DataFrame(X_ca, columns=words_ca)
            crosstab_df['Category'] = df[category_column].astype(str).values
            grouped_ct = crosstab_df.groupby('Category').sum()
            
            # Drop empty categories / never-occurring words: prince needs a non-degenerate table.
            grouped_ct = grouped_ct.loc[grouped_ct.sum(axis=1) > 0, grouped_ct.sum(axis=0) > 0]
            
            if len(grouped_ct) < 2 or grouped_ct.shape[1] < 2:
                st.warning("Perlu minimal dua kategori dan dua kata untuk analisis ini.")
            else:
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
                st.plotly_chart(fig_ca)
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
            X_char = cv_char.fit_transform(df['_clean_text']).toarray()
            words_char = cv_char.get_feature_names_out()
            
            ct = pd.DataFrame(X_char, columns=words_char)
            ct['Group'] = df[category_column].astype(str).values
            ct_sum = ct.groupby('Group').sum()
            
            # chi2_contingency raises on zero row/column totals, which happen whenever a
            # category has no usable tokens (empty cells, all stopwords) or a word never occurs.
            ct_sum = ct_sum.loc[ct_sum.sum(axis=1) > 0, ct_sum.sum(axis=0) > 0]
            
            if len(ct_sum) < 2:
                st.warning("Perlu minimal dua kategori yang berisi teks untuk analisis ini.")
            else:
                # Chi-square test for keyness
                chi2_stats = {}
                for word in words_char:
                    if word not in ct_sum.columns:
                        continue
                    obs = ct_sum[word].values
                    total_words_per_group = ct_sum.sum(axis=1).values - obs
                    contingency_table = np.array([obs, total_words_per_group])
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    chi2_stats[word] = chi2
                    
                keyness_df = pd.DataFrame(list(chi2_stats.items()), columns=['Kata', 'Skor Chi-Square'])
                keyness_df = keyness_df.sort_values(by='Skor Chi-Square', ascending=False)
                
                st.subheader("Kata Paling Signifikan secara Statistik (Chi-Square Keyness)")
                st.dataframe(keyness_df.head(15))
                
                st.subheader("Tabel Silang Frekuensi Kata per Kategori")
                st.dataframe(ct_sum.T)
        else:
            st.info("Pilih 'Kolom Kategori/Bagian' di sidebar sebelah kiri untuk melihat Kata Khas per Bagian.")

else:
    st.info("👋 Silakan unggah file dokumen (CSV, XLS, XLSX, TXT, MD) di sidebar kiri untuk memulai analisis.")