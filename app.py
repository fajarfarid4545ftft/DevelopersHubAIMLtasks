import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    return tokenizer, model

tokenizer, model = load_model()
categories = ["World", "Sports", "Business", "Sci/Tech"]

st.title("News Topic Classifier")
st.write("Type a news headline below to find out its category.")

user_input = st.text_area("Enter News Headline:")

if st.button("Classify Headline"):
    if user_input.strip() != "":
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        scores = outputs.logits
        predicted_class_id = torch.argmax(scores, dim=1).item()
        prediction = categories[predicted_class_id]
        
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please type something first!")
