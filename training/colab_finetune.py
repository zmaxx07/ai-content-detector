# pip install transformers datasets evaluate accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# 1. Load Dataset
dataset = load_dataset("Hello-SimpleAI/HC3", name="all")
# Convert to pairs: 'text' and 'label' (0=Human, 1=AI)
# Assume pre-processing into a standard format here...

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 4. Metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Training Args
# Make sure to update the hub_model_id with your actual Hugging Face username
training_args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    num_train_epochs=3,
    push_to_hub=True,
    hub_model_id="your-username/roberta-ai-detector"
)

# 6. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Ensure your dataset has a 'test' split
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.push_to_hub()
