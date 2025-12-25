## Cats vs Dogs Image Classifier with EfficientNetB0 (TensorFlow/Keras)

<p align="center">
  <img src="https://i.ibb.co/DPkRWnHX/Chat-GPT-Image-Dec-25-2025-02-19-00-PM.jpg" width="100%" />
</p>




In this project, I built an end-to-end image classification pipeline to distinguish **cats vs dogs** using the Kaggle *PetImages* dataset (24,959 images across two classes). I started by validating the dataset structure, inspecting sample images, and setting a reproducible workflow with fixed random seeds. Because the dataset contains a few problematic JPEG files, I made the input pipeline robust by enabling error skipping (`Dataset.ignore_errors()`), and I stabilized training by repeating the datasets and defining explicit `steps_per_epoch`/`validation_steps`.

I trained a **transfer learning** model using **EfficientNetB0** pretrained on ImageNet. The model architecture consisted of the EfficientNetB0 backbone with a lightweight classification head: global average pooling, dropout, and a sigmoid output neuron for binary classification. Training was run on Kaggle GPUs with **mixed precision** enabled for performance. After establishing a strong baseline, I fine-tuned the model by unfreezing the last layers of the backbone with a lower learning rate.

To evaluate performance, I computed confusion matrices, classification reports, and ROC-AUC on the validation split. The model achieved around **99% validation accuracy** with **ROC-AUC ≈ 0.9995**. I then performed **threshold tuning** to find the probability cutoff that maximizes F1-score/accuracy, identifying an optimal threshold of **~0.84**, and demonstrated how changing the threshold shifts the balance between false positives and false negatives. Finally, I saved the trained models (best checkpoint and final version) and created a reusable single-image inference function to classify new images consistently using the chosen threshold.
