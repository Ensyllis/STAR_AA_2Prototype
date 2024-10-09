from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('text', data_files={
    'train': 'C:/Users/josep/Coding/Kliest-Identifying-/STAR-Embedding/seventh_attempt/dataset/train/**/*.txt',
    'test': 'C:/Users/josep/Coding/Kliest-Identifying-/STAR-Embedding/seventh_attempt/dataset/test/**/*.txt'
})

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('AIDA-UPM/star')
model = AutoModelForSequenceClassification.from_pretrained('AIDA-UPM/star', num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer with model, arguments, and dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

# Train the model
trainer.train()

# Evaluate the model after training
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")


# Predict on new texts
new_texts = ["""
    It was just at this time that Gandalf reappeared after his long absence. For three years after the Party he had been away. Then he paid Frodo a brief visit, and after taking a good look at him he went off again. During the next year or two he had turned up fairly often, coming unexpectedly after dusk, and going off without warning before sunrise. He would not discuss his own business and journeys, and seemed chiefly interested in small news about Frodo's health and doings.
""", """
The kitchen door had opened, and there stood Harry's aunt, wearing rubber gloves and a housecoat over her nightdress, clearly halfway through her usual pre-bedtime wipe-down of all the kitchen surfaces. Her rather horsey face registered nothing but shock.
"Albus Dumbledore," said Dumbledore, when Uncle Vernon failed to effect an introduction. "We have corresponded, of course." Harry thought this an odd way of reminding Aunt Petunia that he had once sent her an exploding letter, but Aunt Petunia did not challenge the term. "And this must be your son, Dudley?"
"""]
inputs = tokenizer(new_texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
print(f"Predictions: {predictions}")
