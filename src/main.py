import os
import argparse
from data_loader import load_and_preprocess_data
from model import create_cnn_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN for network intrusion detection.")
    parser.add_argument("--hidden_attack", type=str, default=None, help="Name of the hidden attack type")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classes for classification (2, 6, or 19)")
    parser.add_argument("--max_rows_per_file", type=int, default=None,
                        help="Maximum CSV rows per file for development testing")
    args = parser.parse_args()

    # Get the absolute path of the directory where this script is located 
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Construct the full path to your data directory
    data_dir = os.path.join(script_dir, '..', 'data') 

    # Pass data_dir to the function:
    X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder, test_labels, known_attack_mask = load_and_preprocess_data(
        data_dir, args.class_config, max_rows_per_file=args.max_rows_per_file)  # Pass data_dir here 

    input_shape = (X_train.shape[1], 1) 
    model = create_cnn_model(input_shape, y_train_categorical.shape[1])

    import tensorflow as tf 
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')

    model = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical)

    loss, accuracy = model.evaluate(X_test[known_attack_mask], y_test_categorical)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Uses mask to store only the test rows whose class was visible in the train set.
    # Used for accuracy, precision, recall, f1 score calculations.
    y_test_known = test_labels[known_attack_mask]
    y_pred_known = y_pred[known_attack_mask]

    # Calculates & prints accuracy, precision, recall, f1 scores of attacks visible in the train set.
    accuracy = accuracy_score(y_test_known, y_pred_known)
    precision = precision_score(y_test_known, y_pred_known, average='weighted')
    recall = recall_score(y_test_known, y_pred_known, average='weighted')
    f1 = f1_score(y_test_known, y_pred_known, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test_known, y_pred_known))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_known, y_pred_known))

    # Calculates & prints detection rate, prediction breakdown metrics if an attack was hidden.
    if args.hidden_attack is not None:
        prediction_breakdown = {} # Tracks quantity of classes that model predicted hidden attack as.
        hidden_rows_mask = [] # Stores which rows contain a hidden attack.
        not_benign_count = 0 # Count of how many times model recognizes hidden class as an attack rather than "Benign".
        total_hidden_attacks = 0 # Total number of hidden attack rows.

        for label in test_labels:
            if label == args.hidden_attack:
                hidden_rows_mask.append(True)
            else:
                hidden_rows_mask.append(False)

        for i in range(0, len(test_labels)):
            if hidden_rows_mask[i]:
                # Updates "prediction_breakdown".
                if y_pred[i] in prediction_breakdown:
                    # Adds 1 to count of current class if it exists.
                    prediction_breakdown[y_pred[i]] += 1
                else:
                    # Else adds the class to the dictionary.
                    prediction_breakdown.update({y_pred[i]: 1})

                # Updates count of non-benign classifications.
                if y_pred[i] != "Benign":
                    not_benign_count += 1


        for mask in hidden_rows_mask:
            if mask == True:
                total_hidden_attacks += 1

        # Checks if any attacks were actually hidden. Prevents division by zero.
        if total_hidden_attacks > 0:
            detection_rate = not_benign_count / total_hidden_attacks
            print(f"Detection Rate: {not_benign_count} / {total_hidden_attacks} ({detection_rate:.2%})")
            print("Prediction Breakdown:")

            for class_name in sorted(prediction_breakdown.keys(), reverse = True):
                print(f"\t{class_name}: {prediction_breakdown[class_name]} ({prediction_breakdown[class_name] / total_hidden_attacks:.2%})")
        else:
            print("Warning: Calculations for detection rate and prediction breakdown were skipped. No test rows found for hidden attack:", args.hidden_attack)
