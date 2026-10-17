| ID | Approach | Seeds | Dev macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|---|
| E0 | Majority class (always 'politics') | 1 | 12.5 | 44.8 | 12.4 | 27.8 |
| E1 | TF-IDF words + Complement Naive Bayes | 1 | 79.0 | 85.2 | 81.5 | 84.6 |
| E2 | TF-IDF words + Logistic Regression (baseline) | 1 | 82.2 | 87.0 | 85.2 | 87.1 |
| E3 | TF-IDF char 2-4-grams + Logistic Regression | 1 | 89.0 | 87.4 | 85.6 | 87.4 |
| E4 | TF-IDF words + chars + Linear SVM | 1 | 85.2 | 90.1 | 88.5 | 90.0 |
| E5 | BiLSTM + attention, random embeddings | 3 | 69.9 ± 1.2 | 76.1 ± 0.9 | 73.4 ± 1.1 | 77.5 ± 1.0 |
| E6 | BiLSTM + attention, Word2Vec (frozen) | 3 | 89.7 ± 1.4 | 85.9 ± 1.8 | 84.5 ± 1.6 | 86.3 ± 1.8 |
| E7 | BiLSTM + attention, Word2Vec (fine-tuned) | 3 | 90.6 ± 0.6 | 87.4 ± 1.7 | 85.5 ± 2.1 | 87.6 ± 1.5 |
| E8 | BiGRU + attention, Word2Vec (fine-tuned) | 3 | 90.9 ± 1.4 | 87.4 ± 2.2 | 85.1 ± 2.6 | 87.6 ± 2.2 |
| E9 | XLM-R-base (no Luganda in pretraining) | 3 | 42.8 ± 23.8 | 59.0 ± 13.7 | 40.4 ± 25.0 | 51.8 ± 20.4 |
| E10 | XLM-R-base, stabilised (lr 1e-5, 15 epochs) | 3 | 54.3 ± 7.8 | 67.7 ± 4.7 | 54.9 ± 8.7 | 65.7 ± 5.9 |
| E11 | AfriBERTa-base (no Luganda) | 3 | 87.6 ± 0.4 | 82.4 ± 0.6 | 77.7 ± 0.8 | 82.1 ± 0.6 |
| E12 | AfroXLMR-base (no Luganda) | 3 | 83.8 ± 1.7 | 90.9 ± 0.4 | 89.2 ± 0.6 | 90.8 ± 0.4 |
| E13 | Serengeti (Luganda in pretraining) | 3 | 89.1 ± 2.6 | 86.4 ± 0.8 | 84.4 ± 0.2 | 86.2 ± 0.6 |
| E14 | AfroXLMR-base + our Luganda LAFT | 3 | 87.6 ± 0.7 | 90.1 ± 0.6 | 88.2 ± 0.5 | 90.1 ± 0.5 |
| E15 | AfroXLMR-base + Kirundi/Swahili train data | 3 | 91.6 ± 1.4 | 88.2 ± 1.4 | 85.9 ± 2.4 | 88.0 ± 1.5 |
| E16 | AfroXLMR-base, LoRA r=16 (PEFT) | 3 | 85.9 ± 0.9 | 88.0 ± 0.2 | 86.2 ± 0.3 | 87.8 ± 0.3 |
| E17 | AfroXLMR-base, headline only | 3 | 60.3 ± 2.5 | 72.3 ± 0.9 | 65.8 ± 1.5 | 70.7 ± 1.5 |
