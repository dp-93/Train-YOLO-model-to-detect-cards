# 🃏 Custom Playing Card Detector with YOLOv8

This is a complete, end-to-end computer vision project demonstrating the full lifecycle of training and deploying a custom object detection model. I fine-tuned a state-of-the-art YOLOv8 model to accurately identify and classify 52 different types of playing cards.

The final trained model is deployed as a live, interactive web application on Hugging Face Spaces.

**➡️ [Try the Live Demo Here!](https://huggingface.co/spaces/dp-93/Random_Card_Detector)**

![Live Demo of the Custom Card Detector in Action](link_to_your_gif.gif)

## Project Workflow

This project moves beyond using pre-trained models and focuses on creating a specialized detector for a unique task.

1.  **Dataset Acquisition:** I used a pre-annotated dataset of playing cards from Roboflow, which contains thousands of labeled images necessary for training.

2.  **Model Fine-Tuning (Transfer Learning):** I started with a powerful, pre-trained YOLOv8 model and fine-tuned it on the custom playing card dataset. This allowed the model to leverage its existing knowledge of general object features and quickly become an expert at identifying cards.

3.  **Training & Validation:** The model was trained in a Google Colab environment, achieving an exceptional **mAP50-95 score of over 93%** on the validation set.

4.  **Deployment:** The best-trained model (`best.pt`) was packaged into a Gradio web application and deployed to Hugging Face Spaces, making it publicly accessible for real-time inference.

## Technologies Used

* **AI & Computer Vision:**
    * `YOLOv8`: The core object detection model.
    * `Ultralytics`: The official Python library for training and using YOLO.
    * `Roboflow`: For dataset acquisition and management.

* **Application & Deployment:**
    * `Gradio`: To build the interactive web application.
    * `Hugging Face Spaces`: For hosting the live AI demo.
    * `Python` & `Google Colab`: For the training environment.

## Key Skills Demonstrated

* **Custom Object Detection:** The complete process of fine-tuning a state-of-the-art model for a specific, custom task.
* **Transfer Learning:** A deep, practical understanding of how to leverage pre-trained models.
* **Model Deployment:** Creating a live, interactive application from a trained model file and deploying it on a public platform (Hugging Face).
