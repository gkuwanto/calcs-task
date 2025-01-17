import pandas as pd
import numpy as np


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for particular challenge phase and returns score
    Arguments:
        test_annotations_file: Path to test_annotation_file on the server
        user_submission_file: Path to file submitted by the user
        phase_codename: Phase to which submission is made
        kwargs: Dict of auxiliary params
    Returns:
        Dict with evaluation scores
    """
    try:
        # Read the private leaderboard data
        leaderboard_df = pd.read_csv(test_annotation_file)
        # Read user predictions
        user_predictions = pd.read_csv(user_submission_file)

        # Verify submission format
        expected_columns = ["id", "predicted_choice"]
        if not all(col in user_predictions.columns for col in expected_columns):
            raise Exception(
                "Invalid submission format. Required columns: id, predicted_choice"
            )

        # Calculate accuracy
        correct_predictions = 0
        total_predictions = 0

        for _, row in leaderboard_df.iterrows():
            prediction = user_predictions[user_predictions["id"] == row["id"]][
                "predicted_choice"
            ].iloc[0]
            if prediction == row["chosen"]:
                correct_predictions += 1
            total_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100

        output = {}
        if phase_codename == "dev":
            print("Evaluating for Dev Phase")
            output["result"] = [
                {
                    "train_split": {
                        "Accuracy": accuracy,
                        "Total": accuracy,  # Using accuracy as total score
                    }
                }
            ]
            output["submission_result"] = output["result"][0]["train_split"]

        elif phase_codename == "test":
            print("Evaluating for Test Phase")
            output["result"] = [
                {
                    "test_split": {
                        "Accuracy": accuracy,
                        "Total": accuracy,  # Using accuracy as total score
                    }
                }
            ]
            output["submission_result"] = output["result"][0]

        print(f"Completed evaluation for {phase_codename} Phase")
        return output

    except Exception as e:
        raise Exception(f"Error in evaluation: {str(e)}")
