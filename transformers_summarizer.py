from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Fine-tune the model
def fine_tune_model():
    # Load dataset for fine-tuning
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
    )

    # Load pre-trained model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()

# Summarize text using fine-tuned model
def summarize_text(text, max_words=100, model_name='facebook/bart-large-cnn'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=1024, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Limit summary to maximum number of words
    summary_words = summary.split()[:max_words]
    summary = ' '.join(summary_words)

    return summary

# Fine-tune the model (uncomment the line below to fine-tune)
# fine_tune_model()

# Example usage
