from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import os

def fine_tune_gpt2(data_path, save_dir):
    """
    Fine-tune GPT-2 on the provided solution data.
    Args:
        data_path (str): Path to the solution dataset.
        save_dir (str): Directory to save the fine-tuned model.
    """
    dataset = load_dataset("csv", data_files={"train": data_path})
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    def tokenize_data(batch):
        return tokenizer(batch['text'], truncation=True, padding=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,  # Use mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )
    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Fine-tuned model saved at {save_dir}")

if __name__ == "__main__":
    data_path = "../data/solution.csv"
    save_dir = "../models/fine_tuned_model"
    fine_tune_gpt2(data_path, save_dir)
