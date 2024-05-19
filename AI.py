from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text(text, max_length=150, model_name='facebook/bart-large-cnn'):
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer([text], max_length=max_length, truncation=True, return_tensors='pt')

    # Generate summary
    # Adjust the number of beams for beam search and maximum length of summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=6, max_length=max_length, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    print("Welcome to the Text Summarization Chatbot!")
    while True:
        user_input = input("Please enter the text you want to summarize (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Thank you for using the Text Summarization Chatbot. Goodbye!")
            break
        else:
            summary = summarize_text(user_input)
            print("\nSummary:")
            print(summary)
            print()

if __name__ == "__main__":
    main()
