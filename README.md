## Deep-Learning-Project (NLP)---Text Summarizer

This model is able to provide a summarization from large chunk of texts. The base model used in this project is 't5-small'. It is further trained by 'CNN News 3.0.0' dataset.

## Dataset Description
Please go to the link below for the dataset:
https://huggingface.co/datasets/cnn_dailymail/viewer/3.0.0/train

## The entire model is composed of the following parts:

1. Importing data.
2. Demonstrating some imported dataset.
3. Word cleaning for EDA (Lowering all the letters and eliminating punctuation marks and stopwords)
4. Exploratory Data Analysis (Distribution of words in each article), (Top 10 most appeared words in the articles).
5. Preprocessing dataset (Tokenization).
6. Creating function to calculate 'rouge' score.
7. Defining model parameters.
8. Compiling and training the model.
9. Uploading the model to HuggingFace 
10. Calling and applying the model in actual case.

## Usage:
Please see the demonstration at the bottom of my codes
