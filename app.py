import streamlit as st
import pickle
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Language Detection System",
    page_icon="🌐",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #172554 100%
    );
}


/* Main container */

.block-container {
    max-width: 900px;
    padding-top: 50px;
    padding-bottom: 40px;
}


/* Title */

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}


/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 35px;
}


/* Text area */

.stTextArea textarea {
    background-color: #1f2937 !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}


/* Button */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    border: none;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
    color: white;
}


/* Button hover */

.stButton > button:hover {
    transform: scale(1.01);
    border: none;
}


/* Result box */

.result-box {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
}


/* Result heading */

.result-heading {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 8px;
}


/* Result language */

.result-language {
    color: white;
    font-size: 34px;
    font-weight: 800;
}


/* Section title */

.example-title {
    color: white;
    font-size: 24px;
    font-weight: 700;
    margin-top: 35px;
}


/* Example cards */

.example-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    padding: 15px;
    min-height: 100px;
}


/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Get the folder where app.py is located
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Model path
    model_path = os.path.join(
        base_dir,
        "language_model.pkl"
    )

    # Vectorizer path
    vectorizer_path = os.path.join(
        base_dir,
        "vectorizer.pkl"
    )

    # Check model
    if not os.path.exists(model_path):
        st.error(
            "language_model.pkl not found."
        )
        st.stop()

    # Check vectorizer
    if not os.path.exists(vectorizer_path):
        st.error(
            "vectorizer.pkl not found."
        )
        st.stop()

    # Load model
    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    # Load vectorizer
    with open(
        vectorizer_path,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🌐 Language Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether your text is English, Hindi or Marathi'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TEXT INPUT
# =========================================================

user_text = st.text_area(
    "Enter your text",
    placeholder="Type your sentence here...",
    height=160
)


# =========================================================
# DETECT LANGUAGE
# =========================================================

if st.button("🔍 Detect Language"):

    if user_text.strip() == "":

        st.warning(
            "Please enter some text first."
        )

    else:

        # Convert text into TF-IDF features
        text_tfidf = vectorizer.transform(
            [user_text]
        )

        # Predict language
        prediction = model.predict(
            text_tfidf
        )

        language = prediction[0]

        # Show result
        st.markdown(
            '<div class="result-box">'
            '<div class="result-heading">'
            'Detected Language'
            '</div>'
            f'<div class="result-language">'
            f'{language}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# EXAMPLES
# =========================================================

st.markdown(
    '<div class="example-title">Try Examples</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        '<div class="example-card">'
        '<b style="color:white;">🇬🇧 English</b>'
        '<br><br>'
        '<span style="color:#cbd5e1;">'
        'I am learning Python.'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        '<div class="example-card">'
        '<b style="color:white;">🇮🇳 Hindi</b>'
        '<br><br>'
        '<span style="color:#cbd5e1;">'
        'मुझे पायथन सीखना है।'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        '<div class="example-card">'
        '<b style="color:white;">🇮🇳 Marathi</b>'
        '<br><br>'
        '<span style="color:#cbd5e1;">'
        'मला पायथन शिकायचे आहे.'
        '</span>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    'NLP Project • Character TF-IDF • Logistic Regression • Streamlit'
    '</div>',
    unsafe_allow_html=True
)