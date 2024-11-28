import pandas as pd
from itertools import permutations
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_perplexity(sequence, tokenizer, model):
    """
    Calculate perplexity for a single sequence.
    """
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return torch.exp(torch.tensor(loss)).item()

def beam_search(base_sequence, tokenizer, model, beam_width=5):
    """
    Perform beam search to select the best permutation with minimal perplexity.
    """
    words = base_sequence.split()
    beams = [("", 0)]  # Initialize beams with empty sequence and score

    for word in words:
        new_beams = []
        for seq, score in beams:
            for next_word in words:
                if next_word not in seq.split():  # Avoid duplicate words
                    candidate_seq = f"{seq} {next_word}".strip()
                    perplexity = calculate_perplexity(candidate_seq, tokenizer, model)
                    new_beams.append((candidate_seq, score + perplexity))
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]

    return beams[0][0]  # Return the best sequence

def reorder_text(data_path, model_path, output_path):
    """
    Reorder text in the input data and save results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    data = pd.read_csv(data_path)

    data['text'] = data['text'].apply(lambda row: beam_search(row, tokenizer, model))
    data.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    reorder_text("../data/sample_submission.csv", "../models/fine_tuned_model", "../results/submission.csv")
