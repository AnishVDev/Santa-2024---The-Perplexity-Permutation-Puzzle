# Santa 2024 - Perplexity Minimization Challenge

This project minimizes perplexity for reordered text in the Santa 2024 competition.

## Steps to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Fine-tune the model:
   ```bash
   python src/fine_tune.py

4. Generate submission
   ```bash
   python src/reorder_text.py

6. Validate evaluation locally
   ```bash
   python src/evaluation.py --solution data/sample_submission.csv --submission results/submission.csv
