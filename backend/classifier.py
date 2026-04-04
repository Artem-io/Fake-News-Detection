import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class FakeNewsClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 256

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

        fake_prob = float(probs[0][0])
        real_prob = float(probs[0][1])

        return {
            'prediction': 'REAL' if real_prob > fake_prob else 'FAKE',
            'confidence': max(fake_prob, real_prob),
            'fake_probability': fake_prob,
            'real_probability': real_prob,
        }
