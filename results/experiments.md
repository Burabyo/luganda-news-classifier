| ID | Approach | Seeds | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| E0 | Majority class (always 'politics') | 1 | 12.5 | 44.8 | 12.4 |
| E1 | TF-IDF words + Complement Naive Bayes | 1 | 79.0 | 85.2 | 81.5 |
| E2 | TF-IDF words + Logistic Regression (baseline) | 1 | 82.2 | 87.0 | 85.2 |
| E3 | TF-IDF char 2-4-grams + Logistic Regression | 1 | 89.0 | 87.4 | 85.6 |
| E4 | TF-IDF words + chars + Linear SVM | 1 | 85.2 | 90.1 | 88.5 |
| E5 | BiLSTM + attention, random embeddings | 3 | 69.9 ± 1.2 | 76.1 ± 0.9 | 73.4 ± 1.1 |
| E6 | BiLSTM + attention, Word2Vec (frozen) | 3 | 89.7 ± 1.4 | 85.9 ± 1.8 | 84.5 ± 1.6 |
| E7 | BiLSTM + attention, Word2Vec (fine-tuned) | 3 | 90.6 ± 0.6 | 87.4 ± 1.7 | 85.5 ± 2.1 |
| E8 | BiGRU + attention, Word2Vec (fine-tuned) | 3 | 90.9 ± 1.4 | 87.4 ± 2.2 | 85.1 ± 2.6 |
