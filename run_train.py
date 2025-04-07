
# Import all the necessary libraries as follows:
import pandas as pd
import re
import spacy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Data needs to be preprocessed. Steps include
# Text is converted into lowercase, removes punctuation,digits and extra spaces,Text is being tokenzied and lemmatized using Spacy.

def preprocess_text(text, nlp):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# To handle the tokenized text from the above step dataset class is used, implement to get data by index and length of the dataset.

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

# preprocesses text data and tokenize with BERT,
# handles the class imbalances,
# train BERT model for sequence classification, & save the trained model.

def main():
    nlp = spacy.load("en_core_web_sm")
    data = pd.read_csv('train-1.csv')  # Replace with path to your training data
    data['cleaned_text'] = data['text'].apply(lambda x: preprocess_text(x, nlp))
    data['label'] = data['label'].astype(int)

    # Split data into training and testing sets
    train_texts, _, train_labels, _ = train_test_split(
        data['cleaned_text'], data['label'], test_size=0.2, random_state=42
    )

    # Using BERT's tokenizer to furthur process the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(
        list(train_texts), truncation=True, padding=True, max_length=128, return_tensors="pt"
    )

    # To tackle the class imbalances need to compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', classes=data['label'].unique(), y=data['label']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Set up the DataLoader to help load and shuffle the training data
    train_dataset = TextDataset(train_encodings, train_labels.tolist())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initializa a BERT model for Sequence Classification and define the loss functions for CrossEntropyLoss and chosen AdamW optimizer for training.

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(class_weights))

    # Moving model to correct device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Make sure class_weights are on the correct device
    class_weights = class_weights.to(device)

    # Setting up the loss function and optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    # Run the Model Training Process for 3 Epochs. During this process, the model learns from the data , Computes the gradients, and updates its parameters.
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed.")

    # Finally, Save the trained Model.
    torch.save(model.state_dict(), 'bert_cnn_model.pth')
    print("Model saved as bert_cnn_model.pth")

if __name__ == "__main__":
    main()