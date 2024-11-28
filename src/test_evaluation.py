from evaluation import score
import pandas as pd

if __name__ == "__main__":
    solution_path = "../data/solution.csv"
    submission_path = "../results/submission.csv"
    model_path = "../models/fine_tuned_model"

    solution = pd.read_csv(solution_path)
    submission = pd.read_csv(submission_path)

    try:
        final_score = score(solution, submission, row_id_column_name="id", model_path=model_path)
        print(f"Final perplexity score: {final_score}")
    except Exception as e:
        print(f"Error in scoring: {e}")
