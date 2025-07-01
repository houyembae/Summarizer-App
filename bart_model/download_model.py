# Run download_model.py before using the summarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model.save_pretrained("bart_model")
tokenizer.save_pretrained("bart_model")
