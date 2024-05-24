from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch
 
def fine_tune_model():
    # Load dataset for fine-tuning
    dataset = load_dataset("cnn_dailymail", "3.0.0")
 
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=3e-5,  # Adjusted learning rate
        per_device_train_batch_size=8,  # Increased batch size
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # Increased number of epochs
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,  # Adjusted for increased batch size
        fp16=True,
    )
 
    # Load pre-trained model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
 
    # Metric for evaluation
    rouge_metric = load_metric('rouge')
 
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = ["a" if label == "" else label for label in decoded_labels]
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return result
 
    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
 
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")
 
def summarize_text(text, max_words=150, model_name='facebook/bart-large-cnn'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
 
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
 
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_words,
        num_beams=8,  # Increased number of beams for better quality
        length_penalty=2.0,
        early_stopping=True
    )
 
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
    return summary
 
# Fine-tune the model (uncomment the line below to fine-tune)
# fine_tune_model()
 
# Example usage:
# text = "Your input text here..."
# print(summarize_text(text, max_words=100))