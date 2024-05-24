from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import nltk
from nltk.corpus import wordnet
import random
from itertools import chain
 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
 
def fine_tune_model():
    # Load dataset for fine-tuning
    dataset = load_dataset("cnn_dailymail", "3.0.0")
 
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=200,
        save_total_limit=5,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        fp16=True,
        lr_scheduler_type='linear',
        warmup_steps=500,
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
 
def enrich_text_with_synonyms(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)
    enriched_text = []
 
    # Parts of speech to consider for synonym replacement
    pos_to_replace = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                      'CC', 'IN', 'TO',  # Connectors
                      }
 
    # Iterate through each word and its part of speech
    for word, pos in nltk.pos_tag(words):
        # Check if the part of speech should be replaced
        if pos in pos_to_replace:
            # Get synonyms for the word
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym_words = set(chain.from_iterable([syn.lemma_names() for syn in synonyms]))
                synonym_words.discard(word)
                if synonym_words:
                    # Choose a synonym randomly
                    synonym = random.choice(list(synonym_words))
                    # Ensure synonym fits naturally into the sentence
                    enriched_text.append(synonym.replace('_', ' '))
                    continue  # Move to the next word
        enriched_text.append(word)
 
    # Join the enriched words to form the final text
    return ' '.join(enriched_text)
 
def summarize_text(text, model_name='facebook/bart-large-cnn'):
    # Initialize tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
 
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
    input_length = inputs['input_ids'].shape[1]
 
    # Determine max_length dynamically
    max_length = dynamic_max_length(input_length)
 
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_beams=10,
        length_penalty=2.0,
        early_stopping=True
    )
 
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
    # Enrich summary with synonyms
    enriched_summary = enrich_text_with_synonyms(summary)
 
    return enriched_summary
 
def dynamic_max_length(input_length):
    if input_length < 500:
        return 300
    elif input_length < 1000:
        return 400
    else:
        return 500
 
# Example usage:
# text = "Your input text here..."
# print(summarize_text(text))