import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# ---------------------------------------------------------
# 1. إعدادات الصفحة و NLTK
# ---------------------------------------------------------
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

@st.cache_resource
def download_nltk_data():
    packages = [
        'punkt',
        'punkt_tab',
        'stopwords', 
        'wordnet', 
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng', 
        'omw-1.4'
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            pass
download_nltk_data()

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

# ---------------------------------------------------------
# 2. تعريف طبقة الـ Attention الخاصة بموديل Keras
# ---------------------------------------------------------
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                  initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                  initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

# ---------------------------------------------------------
# 3. تنظيف النصوص
# ---------------------------------------------------------
def get_wordnet_pos(treebank_tag: str) -> str:
    mapping = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
    return mapping.get(treebank_tag[0], wordnet.NOUN)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = text.split()
    pos_tags = nltk.pos_tag(words)
    return ' '.join(
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and len(word) > 1
    )

# ---------------------------------------------------------
# 4. تحميل الموديلات (بما فيها الـ Label Encoder للدومين)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        svm_model = joblib.load('spam_svm_model.joblib')
        xgb_model = joblib.load('spam_xgb_model.joblib')
        tfidf = joblib.load('tfidf_vectorizer.joblib')
        scaler = joblib.load('scaler.joblib')
        dl_tokenizer = joblib.load('keras_tokenizer.joblib')
        label_encoder = joblib.load('label_encoder.joblib')  # تحميل ملف تشفير الدومين
        dl_model = load_model('spam_hybrid_attention_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
        return svm_model, xgb_model, dl_model, tfidf, scaler, dl_tokenizer, label_encoder
    except Exception as e:
        return None, None, None, None, None, None, None

svm_model, xgb_model, dl_model, tfidf, scaler, dl_tokenizer, label_encoder = load_assets()

# ---------------------------------------------------------
# 5. واجهة المستخدم (UI)
# ---------------------------------------------------------
st.title("📧 Spam Email Classifier App")
st.write("تطبيق بسيط لتصنيف الإيميلات (سبام أو سليم).")

st.markdown("---")

col_perf, col_features = st.columns([1, 1])

with col_perf:
    st.subheader("📊 أداء الموديلات (Evaluation Metrics)")
    st.write("نتائج الاختبار على الـ Test Data:")
    
    metrics_data = {
        "الموديل (Model)": ["Linear SVM", "XGBoost", "BiLSTM + Attention"],
        "Accuracy": ["100%", "99.95%", "100%"],
        "Precision": ["100%", "100%", "100%"],
        "Recall": ["100%", "100%", "100%"],
        "F1-Score": ["100%", "100%", "100%"]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)

with col_features:
    st.subheader("💡 إيه اللي بيأثر على قرارات الموديل؟")
    st.write("الموديل مش بيقرأ النص بس، لكنه بيعتمد على **نص الإيميل + 12 ميزة إضافية (Columns)**:")
    with st.expander("📌 اضغط هنا عشان تشوف الأعمدة وتأثيرها", expanded=False):
        st.markdown("""
        **1. النص (Text Features):**
        * الكلمات المستخدمة (هل فيها كلمات احتيال زي Free, Money, Urgent؟).
        
        **2. ميزات رقمية (Metadata Columns):**
        * **num_words**: عدد الكلمات في الإيميل.
        * **num_links**: عدد اللينكات اللي جوه الرسالة.
        * **has_suspicious_link**: هل اللينك مشبوه؟
        * **sender_reputation_score**: درجة موثوقية المرسل (سمعته).
        * **num_attachments**: عدد المرفقات (Attachments).
        * **contains_money_terms / urgency_terms**: هل بيحتوي على صيغة طلب فلوس أو استعجال؟
        * **email_hour / is_weekend**: وقت الإرسال وهل هو في إجازة الأسبوع ولا لأ.
        * **num_recipients**: عدد الأشخاص اللي مبعوتلهم الإيميل.
        * **sender_domain**: دومين المرسل (زي gmail, yahoo أو دومين مشبوه).
        """)

st.markdown("---")

st.subheader("🧪 اختبر الموديل بنفسك")
model_choice = st.selectbox("اختار الموديل اللي عايز تجربه:", 
                            ["Linear SVM", "XGBoost", "BiLSTM + Attention"])

# إضافة حقل إيميل المرسل
sender_email = st.text_input("إيميل المرسل (Sender Email):", placeholder="example@gmail.com")
email_subject = st.text_input("موضوع الإيميل (Subject):", placeholder="Congratulations! You won...")
email_body = st.text_area("نص الإيميل (Body):", height=150, placeholder="Click here to claim your cash now...")

if st.button("🚀 افحص الإيميل", type="primary"):
    email_input = email_subject + " " + email_body
    
    if not email_input.strip():
        st.warning("أرجوك اكتب موضوع أو نص الإيميل الأول عشان نقدر نفحصه.")
    elif svm_model is None:
        st.error("مش قادر ألاقي ملفات الموديل! اتأكدي إن ملفات الـ joblib والـ keras في نفس الفولدر.")
    else:
        with st.spinner('جاري الفحص...'):
            clean_email = preprocess_text(email_input)
            
            # --- معالجة دومين المرسل (Domain Processing) ---
            domain_encoded = 0
            if label_encoder is not None and sender_email.strip():
                # استخراج الدومين (اللي بعد علامة @)
                domain = sender_email.split('@')[-1].lower().strip()
                # التأكد إن الدومين موجود في بيانات التدريب
                if domain in label_encoder.classes_:
                    domain_encoded = label_encoder.transform([domain])[0]
            
            # تجهيز الميزات الرقمية الوهمية عشان الموديل يشتغل
            num_words = len(clean_email.split())
            contains_money = 1 if any(word in clean_email for word in ['money', 'cash', 'dollar', 'free', 'win']) else 0
            contains_urgency = 1 if any(word in clean_email for word in ['urgent', 'now', 'immediate', 'offer', 'claim']) else 0
            
            # إضافة الـ domain_encoded في آخر المصفوفة بدال الصفر الثابت
            numeric_features = np.array([[num_words, 0, 0, 0, 0.5, 12, 3, 0, 1, contains_money, contains_urgency, domain_encoded]])
            numeric_scaled = scaler.transform(numeric_features)
            
            prediction = 0
            
            if model_choice in ["Linear SVM", "XGBoost"]:
                text_tfidf = tfidf.transform([clean_email])
                X_combined = csr_matrix(hstack((text_tfidf, numeric_scaled)))
                
                if model_choice == "Linear SVM":
                    prediction = svm_model.predict(X_combined)[0]
                else:
                    prediction = xgb_model.predict(X_combined)[0]
                    
            elif model_choice == "BiLSTM + Attention":
                text_seq = dl_tokenizer.texts_to_sequences([clean_email])
                text_pad = pad_sequences(text_seq, maxlen=150, padding='post', truncating='post')
                
                prob = dl_model.predict([text_pad, numeric_scaled])[0][0]
                prediction = 1 if prob > 0.5 else 0

            # عرض النتيجة
            st.markdown("### النتيجة:")
            if prediction == 1:
                st.error("🚨 **النتيجة:** ده إيميل احتيالي / سبام (Spam)")
            else:
                st.success("✅ **النتيجة:** ده إيميل سليم (Not Spam / Ham)")