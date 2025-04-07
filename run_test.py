# Import all the necessary libraries as follows: 
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# To handle tokenized text and labels, and retrieving samples and the dataset's size.
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def main():
    # Loading the BERT tokenizer and model to classify text sequences.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('bert_cnn_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loading the test data from a CSV file.
    test_data = pd.read_csv('/Users/sripadma/Desktop/ML- Project/train-1.csv')  # Replace with path to test data
    test_encodings = tokenizer(
        list(test_data['text']), truncation=True, padding=True, max_length=128, return_tensors="pt"
    )

    test_dataset = TextDataset(test_encodings, [0]*len(test_data))  
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Changing the model to evaluate and making predictionsthrough iterate over the DataLoader without computing gradients.
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    # Adding a new column for predictions to the test data. Saving the test data with predictions into a CSV file.
    test_data['predictions'] = predictions
    test_data.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

    # If the true labels are available in the test data, computing the  evaluation metrics like accuracy, precision, recall, and F1-score. 
    # Priting the classification report for a detailed performance test.
    if 'label' in test_data.columns:
        accuracy = accuracy_score(test_data['label'], predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_data['label'], predictions, average='macro')
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_data['label'], predictions))

if __name__ == "__main__":
    main()
