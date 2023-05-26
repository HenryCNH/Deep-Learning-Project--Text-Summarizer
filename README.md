## Deep-Learning-Project (NLP)---Text Summarizer

This model is able to provide a summarization from an article. The base model used in this project is 't5-small'. It is further trained by 'CNN News 3.0.0' dataset.

## Dataset Description
Please go to the link below to download the dataset:
https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0/train

## The entire model is composed of the following parts:

1. Importing data.
2. Demonstrating some imported dataset.
3. Exploratory Data Analysis (Distribution of words in each article), (Top 10 most appeared words in the articles).
4. Preprocessing dataset (Tokenization).
5. Creating function to calculate 'rouge score.
6. Defining model parameters.
7. Compiling and training the model.
8. Cloning the model to HuggingFace 
9. Calling and applying the model in actual case.
