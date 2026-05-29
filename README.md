# 📧 Spam Email Classifier — *Because Your Inbox Deserves Better*

## 🌐 Live Demo

**Don't want to run it locally? We got you.** 👇

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bainamagdy-nlp-project-app-pw8j7h.streamlit.app/)

> 🔗 **[https://bainamagdy-nlp-project-app-pw8j7h.streamlit.app/](https://bainamagdy-nlp-project-app-pw8j7h.streamlit.app/)**

Just paste a suspicious email and let the models do the dirty work. No setup needed. ✨

---

> "Dear Winner, You Have Been Selected To Claim Your $10,000,000 Prize!!!"
> 
> — Every spam email ever. We built a machine to destroy them. 🔫

---

## 🤔 What Is This?

You know that feeling when you open your inbox and it's 90% Nigerian princes, fake lottery wins, and "URGENT: Your account has been compromised" emails?

**Yeah. We fixed that.**

This project is a **Spam Email Classifier** powered by three machine learning models working together like a very well-trained, very suspicious bouncer at the door of your inbox.

---

## 🧠 The Brains Behind the Operation

We didn't just throw *one* model at this problem. We threw **three**.

| Model | Vibe | Accuracy |
|---|---|---|
| 🗡️ **Linear SVM** | The classic. Fast, reliable, no drama. | **100%** |
| 🚀 **XGBoost** | The overachiever who still shows up | **99.95%** |
| 🧬 **BiLSTM + Attention** | The nerd who reads every single word *and* pays attention | **100%** |

> Yes, those numbers are real. No, we are not sorry. 😎

---

## 🕵️ What Does the Model Actually Look At?

Your email doesn't just get read — it gets **interrogated**.

**Text Analysis:**
- Every word is lemmatized, cleaned, and TF-IDF vectorized with bigrams (because "free money" hits different than "free" and "money" separately)

**12 Metadata Features (a.k.a. The Suspicious Checklist™):**

| Feature | Why We Care |
|---|---|
| `num_words` | Spammers love to ramble |
| `num_links` | 17 links in one email? Suspicious. |
| `has_suspicious_link` | If it looks sketchy, it probably is |
| `sender_reputation_score` | Is this sender trusted or a ghost? |
| `num_attachments` | "Open this zip file" — no thanks |
| `contains_money_terms` | Free! Cash! Win! Dollar! 🚨 |
| `contains_urgency_terms` | ACT NOW! LIMITED TIME! CLAIM TODAY! |
| `email_hour` | Who sends business emails at 3 AM? |
| `is_weekend` | Real companies don't panic on Sundays |
| `num_recipients` | Sent to 847 people? Red flag. |
| `sender_domain` | gmail ✅ / xn--free-prize-4u.ru ❌ |

---

## 🏗️ Project Structure

```
📦 spam-email-classifier
 ┣ 📜 app.py                          ← Streamlit web app (the pretty face)
 ┣ 📓 spam_classifier_final.ipynb     ← Where all the magic was cooked
 ┣ 🧠 spam_svm_model.joblib           ← Trained SVM
 ┣ 🧠 spam_xgb_model.joblib           ← Trained XGBoost
 ┣ 🧠 spam_hybrid_attention_model.keras ← Trained BiLSTM + Attention
 ┣ 🔤 tfidf_vectorizer.joblib         ← Text vectorizer
 ┣ ⚖️  scaler.joblib                  ← Feature scaler
 ┣ 🔡 keras_tokenizer.joblib          ← Tokenizer for deep learning
 ┣ 🏷️  label_encoder.joblib           ← Sender domain encoder
 ┗ 📄 README.md                       ← You are here 👋
```

---

## 🛠️ Tech Stack

```python
languages  = ["Python 3.10+"]
ml         = ["scikit-learn", "XGBoost", "TensorFlow / Keras"]
nlp        = ["NLTK", "TF-IDF", "WordNet Lemmatizer", "POS Tagging"]
app        = ["Streamlit"]
data       = ["pandas", "NumPy", "SciPy"]
storage    = ["joblib"]
```

---

## 🚀 How To Run It Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/bainamagdy/NLP-Project.git
cd NLP-Project
```

**Step 2 — Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn xgboost tensorflow nltk joblib scipy
```

**Step 3 — Launch the app**
```bash
streamlit run app.py
```

**Step 4 — Paste a sketchy email and watch the model destroy it** 🔥

---

## 🖥️ The App (Streamlit UI)

The web app lets you:
- Choose between the **3 trained models**
- Enter the **sender's email**, **subject**, and **body**
- Hit **"Check Email"** and get an instant verdict

**Spam detected:**
> 🚨 This email is Spam / Fraudulent.

**Legit email:**
> ✅ This email is Legitimate (Ham).

---

## 🔬 How We Trained This Thing

1. **Loaded** a structured CSV dataset with email text + metadata
2. **Audited for feature leakage** (because inflated accuracy is a lie and we don't do that here)
3. **Dropped** useless columns: `email_id`, `has_attachment`, `num_characters`, `num_exclamation_marks`
4. **Preprocessed text**: lowercase → remove punctuation/digits → POS-aware lemmatization → stopword removal
5. **Encoded** `sender_domain` with LabelEncoder (fitted on train only — no leakage!)
6. **Vectorized** text with TF-IDF (3,000 features, bigrams, min_df=2)
7. **Scaled** numeric features with StandardScaler
8. **Trained** all 3 models on the combined feature matrix
9. **Evaluated** with Accuracy, Precision, Recall, F1-Score + 5-fold Cross-Validation
10. **Saved** everything cleanly so the app just works™

---

## ⚔️ Challenges We Fought (And Won)

| Challenge | How We Beat It |
|---|---|
| Feature leakage from metadata | Ran a leakage audit *before* training |
| Unknown sender domains at inference | Map unseen domains safely to `0` |
| Class imbalance | Used `stratify=y` in train/test split |
| BiLSTM overfitting | Early stopping with `patience=3` |
| LinearSVC has no `predict_proba` | Wrapped with `CalibratedClassifierCV` |

---

## 📚 References

This project stands on the shoulders of giants (and a lot of Stack Overflow answers):

- Almeida & Hidalgo (2012) — SMS Spam Filtering
- Chen & Guestrin (2016) — XGBoost
- Hochreiter & Schmidhuber (1997) — LSTM
- Vaswani et al. (2017) — Attention Is All You Need
- Pedregosa et al. (2011) — scikit-learn
- Manning et al. (2008) — Introduction to Information Retrieval
- Cortes & Vapnik (1995) — Support Vector Networks
- The entire NLTK documentation (bless you all 🙏)

---

## 👩‍💻 The Dream Team

Built with ☕, frustration, and an unreasonable hatred of spam emails by four people who now read every email with extreme suspicion:

| | Name |
|---|---|
| 🌸 | **Shahd Ramadan** |
| 🌻 | **Baina Magdy** |
| 🌺 | **Nourhan Mohammed** |
| 🌼 | **Israa Alaa** |

> *"Four girls. Three models. Zero spam. No regrets."*
>
> — The Team, probably at 2 AM before the deadline

> *"If you got this far in the README, you're clearly not a spam bot."*
> 
> — Us, right now

---

⭐ **[Star this repo](https://github.com/bainamagdy/NLP-Project)** if it helped you! Your star means we didn't do this for nothing. 😄
