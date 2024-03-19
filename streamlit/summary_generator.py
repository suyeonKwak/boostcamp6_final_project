from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

nltk.download("punkt")


def build_text_model():
    model_dir = "lcw99/t5-large-korean-text-summary"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer


def summarize_text(text: str):

    model, tokenizer = build_text_model()

    max_input_length = 512 + 256

    inputs = ["summarize: " + text]

    inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
    )
    output = model.generate(
        **inputs, num_beams=8, do_sample=True, min_length=10, max_length=30
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    return predicted_title
