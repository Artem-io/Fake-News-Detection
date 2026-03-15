import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from lime.lime_text import LimeTextExplainer


class FakeNewsClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 256
        self.explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Batch prediction returning probabilities — used by LIME internally."""
        all_probs = []
        for text in texts:
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

            all_probs.append(probs[0].cpu().numpy())
        return np.array(all_probs)

    def predict(self, text: str) -> dict:
        probs = self._predict_proba([text])[0]
        fake_prob = float(probs[0])
        real_prob = float(probs[1])

        return {
            'prediction': 'REAL' if real_prob > fake_prob else 'FAKE',
            'confidence': max(fake_prob, real_prob),
            'fake_probability': fake_prob,
            'real_probability': real_prob
        }

    def explain(self, text: str, num_features: int = 15, num_samples: int = 200) -> dict:
        """
        Run LIME to explain which words push the model toward FAKE vs REAL.

        Returns dict with:
          - prediction, confidence, fake/real probabilities
          - lime_explanation: list of {word, weight, direction}
            positive weight → pushes toward predicted class
            negative weight → pushes away from predicted class
        """
        prediction = self.predict(text)
        predicted_label = 0 if prediction['prediction'] == 'FAKE' else 1

        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(predicted_label,),
        )

        word_weights = explanation.as_list(label=predicted_label)

        lime_words = []
        for word, weight in word_weights:
            lime_words.append({
                "word": word,
                "weight": round(weight, 4),
                "direction": prediction['prediction'] if weight > 0 else (
                    "REAL" if prediction['prediction'] == "FAKE" else "FAKE"
                ),
            })

        return {
            **prediction,
            "lime_explanation": lime_words,
        }
