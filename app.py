import string
import math
from collections import Counter
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. DOWNLOAD NLTK DATA SILENTLY ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. FAQ DATASET ---
FAQ_DATA = [
    {
        "question": "What is the return policy?",
        "answer": "You can return any unused item within 30 days of delivery for a 100% refund."
    },
    {
        "question": "How can I track my order status?",
        "answer": "Head over to 'My Orders' section in your profile to track your package live."
    },
    {
        "question": "What payment methods are available?",
        "answer": "We accept UPI (Google Pay, PhonePe, Paytm), Credit/Debit cards, Net Banking, and COD."
    },
    {
        "question": "How long does shipping and delivery take?",
        "answer": "Standard delivery takes 3 to 5 business days. Express shipping delivers within 24-48 hours."
    },
    {
        "question": "How do I cancel my order?",
        "answer": "You can cancel your order directly from the orders page before it gets dispatched."
    },
    {
        "question": "How can I contact customer care support?",
        "answer": "You can email us at support@store.com or reach out via toll-free helpline 1800-000-9999."
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Currently, we only deliver across India. International shipping will be available soon."
    }
]

# --- 3. NLP PREPROCESSING & MATCHING ---
def preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    return clean_tokens

def calculate_cosine_similarity(tokens1, tokens2):
    """Pure Python Cosine Similarity (no scikit-learn needed)"""
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([val ** 2 for val in vec1.values()])
    sum2 = sum([val ** 2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

# Pre-tokenize all FAQs
faq_processed = [preprocess(item["question"]) for item in FAQ_DATA]

def get_answer(user_query, threshold=0.20):
    user_tokens = preprocess(user_query)
    
    if not user_tokens:
        return "Please ask a specific question so I can help you better!"
        
    scores = [
        calculate_cosine_similarity(user_tokens, faq_tokens)
        for faq_tokens in faq_processed
    ]
    
    best_score = max(scores)
    best_index = scores.index(best_score)
    
    if best_score < threshold:
        return "I'm sorry, I don't have information on that yet. Try asking about order tracking, returns, shipping, or payments!"
    
    return FAQ_DATA[best_index]["answer"]

# --- 4. STREAMLIT CHAT UI ---
st.set_page_config(page_title="FAQ Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Customer Support FAQ Bot")
st.caption("Ask me questions about returns, orders, delivery, and payments.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    bot_reply = get_answer(user_prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)