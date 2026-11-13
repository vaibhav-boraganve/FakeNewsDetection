# Model weights

Drop trained weights here; each detector picks them up automatically on startup.

| File / folder | Produced by | Used by |
|---|---|---|
| `text_tfidf.joblib` (+ `text_tfidf.metrics.json`) | auto-trained on first start, or `python -m training.train_text` | Text – TF-IDF + Logistic Regression |
| `text_transformer/` (HF `save_pretrained` folder) | `python -m training.train_text --transformer bert-base-uncased` | Text – fine-tuned BERT (overrides the hub model) |
| `image_efficientnet_b0.pt` | `python -m training.train_image --arch efficientnet` | Image – EfficientNet-B0 |
| `xception_faces.pt` | `python -m training.train_image --arch xception --faces` (or FaceForensics++ Xception checkpoint) | Image faces + video frames – XceptionNet |
| `video_cnn_lstm.pt` | `python -m training.train_video` | Video – CNN + BiLSTM |

Checkpoints may be plain `state_dict`s, `{"state_dict": ...}` dicts, or DataParallel-wrapped;
`module.` / `model.` prefixes and the FaceForensics++ `last_linear` head are remapped automatically.
