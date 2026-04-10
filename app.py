# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import re

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Allergen Detector",
    page_icon="🍽️",
    layout="centered"
)

# ── Load Pipeline ─────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    model             = joblib.load('pipeline/model.pkl')
    tfidf             = joblib.load('pipeline/tfidf.pkl')
    mlb               = joblib.load('pipeline/mlb.pkl')
    category_columns  = joblib.load('pipeline/category_columns.pkl')
    allergen_tags     = joblib.load('pipeline/allergen_tags.pkl')
    allergen_keywords = joblib.load('pipeline/allergen_keywords.pkl')
    return model, tfidf, mlb, category_columns, allergen_tags, allergen_keywords

model, tfidf, mlb, category_columns, allergen_tags, allergen_keywords = load_pipeline()

RARE_ALLERGENS = [
    'en:lupin',
    'en:molluscs',
    'en:sulphur-dioxide-and-sulphites'
]

ALLERGEN_EMOJI = {
    'gluten':                        '🌾',
    'crustaceans':                   '🦐',
    'eggs':                          '🥚',
    'fish':                          '🐟',
    'peanuts':                       '🥜',
    'soybeans':                      '🫘',
    'milk':                          '🥛',
    'nuts':                          '🌰',
    'celery':                        '🥬',
    'mustard':                       '🌿',
    'sesame-seeds':                  '🌱',
    'sulphur-dioxide-and-sulphites': '⚗️',
    'lupin':                         '🌼',
    'molluscs':                      '🐚',
}

# ── Prediction Function ───────────────────────────────────────────────
def predict_allergens(ingredients_text, category='unknown'):
    # Clean text
    clean = ingredients_text.lower()
    clean = re.sub(r'[^a-z0-9\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # TF-IDF
    X_tfidf = tfidf.transform([clean])

    # Category encoding
    sample_cat = pd.get_dummies(pd.Series([category]), prefix='cat')
    sample_cat = sample_cat.reindex(columns=category_columns, fill_value=0).values.astype(np.float32)

    # Keyword flags
    flags = []
    for tag, keywords in allergen_keywords.items():
        flag = 1 if any(kw in clean for kw in keywords) else 0
        flags.append(flag)
    X_flags = np.array(flags, dtype=np.float32).reshape(1, -1)

    # Ingredient count
    X_count = np.array([[len(clean.split(','))]], dtype=np.float32)

    # Combine
    X = sp.hstack([
        X_tfidf,
        sp.csr_matrix(sample_cat),
        sp.csr_matrix(X_flags),
        sp.csr_matrix(X_count)
    ], format='csr')

    # Predict with threshold adjustment
    probs = np.array([est.predict_proba(X)[:, 1] for est in model.estimators_]).T
    results = []
    for i, tag in enumerate(allergen_tags):
        threshold = 0.3 if tag in RARE_ALLERGENS else 0.5
        detected  = probs[0][i] >= threshold
        confidence = probs[0][i]
        uncertain  = 0.3 <= confidence < 0.8
        results.append({
            'tag':        tag,
            'label':      tag.replace('en:', ''),
            'detected':   bool(detected),
            'confidence': round(float(confidence) * 100, 1),
            'uncertain':  bool(uncertain and detected)
        })
    return results

# ── UI ────────────────────────────────────────────────────────────────
st.title("🍽️ Food Allergen Detection System")
st.markdown("#### Identify the 14 UK regulated allergens from ingredient text")
st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    ingredients_input = st.text_area(
        "Paste ingredient list here:",
        placeholder="e.g. Wheat flour, milk, butter, eggs, sugar, salt, yeast...",
        height=150
    )
with col2:
    category = st.selectbox(
        "Product category:",
        options=[
            'unknown', 'Sugary snacks', 'Cereals and potatoes',
            'Milk and dairy products', 'Fish Meat Eggs',
            'Fat and sauces', 'Salty snacks', 'Beverages',
            'Fruits and vegetables', 'Composite foods'
        ]
    )
    st.markdown("<br>", unsafe_allow_html=True)
    detect_btn = st.button("🔍 Detect Allergens", use_container_width=True)

# Disclaimer
st.warning("⚠️ This tool supplements but does **not replace** reading the manufacturer's label. Always verify allergen information independently.")

# ── Results ───────────────────────────────────────────────────────────
if detect_btn:
    if not ingredients_input.strip():
        st.error("Please enter ingredient text before detecting.")
    else:
        with st.spinner("Analysing ingredients..."):
            results = predict_allergens(ingredients_input, category)

        detected   = [r for r in results if r['detected']]
        undetected = [r for r in results if not r['detected']]

        st.markdown("---")

        if detected:
            st.error(f"### ⚠️ {len(detected)} Allergen(s) Detected")
            cols = st.columns(3)
            for idx, r in enumerate(detected):
                with cols[idx % 3]:
                    emoji = ALLERGEN_EMOJI.get(r['label'], '⚠️')
                    badge = " ⚠️ *uncertain*" if r['uncertain'] else ""
                    color = "orange" if r['uncertain'] else "red"
                    st.markdown(
                        f"""
                        <div style='background:#fff0f0;border-left:4px solid {color};
                                    padding:10px;border-radius:6px;margin-bottom:8px'>
                            <b>{emoji} {r['label'].replace('-', ' ').title()}</b>{badge}<br>
                            <small>Confidence: {r['confidence']}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.success("### ✅ No Allergens Detected")
            st.info("No regulated allergens were identified. Always verify with the product label.")

        # Safe allergens
        with st.expander("✅ View allergens NOT detected"):
            cols = st.columns(4)
            for idx, r in enumerate(undetected):
                with cols[idx % 4]:
                    emoji = ALLERGEN_EMOJI.get(r['label'], '✅')
                    st.markdown(f"<small>{emoji} {r['label'].replace('-',' ').title()}</small>",
                                unsafe_allow_html=True)

        # Confidence table
        with st.expander("📊 View full confidence scores"):
            conf_df = pd.DataFrame([{
                'Allergen':    r['label'].replace('-', ' ').title(),
                'Detected':    '✅ Yes' if r['detected'] else '❌ No',
                'Confidence':  f"{r['confidence']}%",
            } for r in results])
            st.dataframe(conf_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>COM763 Advanced Machine Learning · Wrexham University · "
    "Data: Open Food Facts (ODbL)</small></center>",
    unsafe_allow_html=True
)