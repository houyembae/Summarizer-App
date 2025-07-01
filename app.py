import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model()


st.title("Text Summarization with BART")


input_text = st.text_area("Enter a long paragraph or article:", height=1024)

# Generate summary
if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=142,
            min_length=56,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary")
        st.success(summary)
