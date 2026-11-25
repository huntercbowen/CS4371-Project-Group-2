import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from data_loader import load_and_preprocess_data
from model import create_cnn_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def create_bar_chart_output(hidden_attack: str, prediction_breakdown, script_dir) -> None:
    """Create prediction breakdown bar chart and add to graphs directory."""

    # Gets class names and the count of how many times they were each predicted by the model.
    class_names = list(prediction_breakdown.keys()) 
    values = list(prediction_breakdown.values())

    # Creates bar graph.
    plt.figure(figsize=(9,5))
    bar_info = plt.barh(class_names, values,)
    plt.bar_label(bar_info, padding = 2) # Adds labels to each bar.
    plt.xlabel('Number of Predictions')
    plt.ylabel('Predicted Attack Type')
    plt.subplots_adjust(left=0.30) # Increases whitespace margin for class names.
    plt.title(f'Prediction breakdown for {hidden_attack}')
    plt.xlim(0, max(values) * 1.10) # Extends x-axis to correctly show bars without overlap.

    # Creates the PNG image output.
    date_time = datetime.now().strftime('%Y-%m-%d_%H-%M') # Gets current date and time for file name.
    graph_folder = os.path.join(script_dir, '..', 'results', 'graphs')
    os.makedirs(graph_folder, exist_ok=True) # Makes the graphs dir if it does not already exist.
    file_name = f'{args.hidden_attack}_{date_time}.png'
    #file_name = f'{date_time}.png'
    file_path = os.path.join(graph_folder, file_name)
    plt.savefig(file_path, dpi = 300, bbox_inches = 'tight') # Saves the chart as a PNG image.
    print(f'Image {file_name} was saved to {graph_folder}')
    plt.show()

def create_confusion_matrix_output(class_names, confusion_matrix_values) -> None:
    """Create confusion matrix output and add to graphs directory."""

    plt.figure(figsize=(max(8, len(class_names) * 0.7), max(6, len(class_names) * 0.6)))
    image = plt.imshow(confusion_matrix_values, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(image)
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    threshold = confusion_matrix_values.max() / 2

    for row in range(confusion_matrix_values.shape[0]):
        for column in range(confusion_matrix_values.shape[1]):
            value = confusion_matrix_values[row, column]
            plt.text(
                column,
                row,
                format(value, "d"),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    graph_folder = os.path.join(script_dir, "..", "results", "graphs")
    os.makedirs(graph_folder, exist_ok=True)
    file_name = f"confusion_matrix_{date_time}.png"
    file_path = os.path.join(graph_folder, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix image {file_name} was saved to {graph_folder}")
    plt.show()
    plt.close()

def create_csv_output(script_dir: str) -> None:
    """Update metrics.csv with calculated metrics of each run."""

    # Makes the CSV dir within data if it does not already exist.
    csv_folder = os.path.join(script_dir, '..', 'results', 'csv')
    os.makedirs(csv_folder, exist_ok=True)

    csv_path = os.path.join(csv_folder, 'metrics.csv')
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                'Timestamp',
                'Hidden Attack',
                'Class Config',
                'Accuracy',
                'Precision',
                'Recall',
                'F1 Score',
                'Detection Rate'
            ])

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            args.hidden_attack if args.hidden_attack is not None else 'None',
            args.class_config,
            accuracy,
            precision,
            recall,
            f1,
            detection_rate
        ])

    print(f"Metrics successfully saved to {csv_path}")

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

    # Creates & prints classification report and confusion matrix.
    class_names = label_encoder.classes_
    confusion_matrix_values = confusion_matrix(
        y_test_known,
        y_pred_known,
        labels=class_names,
    )

    print("\nClassification Report:\n", classification_report(y_test_known, y_pred_known))
    print("\nConfusion Matrix:\n", confusion_matrix_values)

    # Creates the confusion matrix output.
    create_confusion_matrix_output(class_names, confusion_matrix_values)

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

            for class_name in sorted(prediction_breakdown.keys()):
                print(f"\t{class_name}: {prediction_breakdown[class_name]} ({prediction_breakdown[class_name] / total_hidden_attacks:.2%})")

            # Creates the bar chart output for the prediction breakdown.
            create_bar_chart_output(args.hidden_attack, prediction_breakdown, script_dir)

            # Creates the csv output.
            create_csv_output(script_dir)
        else:
            print("Warning: Calculations for detection rate and prediction breakdown were skipped. No test rows found for hidden attack:", args.hidden_attack)
