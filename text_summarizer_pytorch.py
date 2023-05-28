import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import logging
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from transformers.keras_callbacks import KerasMetricCallback
from transformers import pipeline
from nlp_utils import *
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
from transformers import TFAutoModelForSeq2SeqLM
from transformers import pipeline
from transformers.utils import send_example_telemetry
import nltk
nltk.download('punkt')

send_example_telemetry("summarization_notebook", framework="pytorch")

from huggingface_hub import notebook_login

notebook_login()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Importing Dataset
dataset_train=load_dataset("cnn_dailymail", '3.0.0', split='validation')
dataset_test=load_dataset("cnn_dailymail", '3.0.0', split='test')
print(dataset_train)

tf.debugging.set_log_device_placement(True)
#Exploratory Data Analysis for training dataset
df_train=pd.DataFrame(dataset_train)
df_train.info()
word_count=[]
print(df_train.head())


#EDA - Exploring word data
for i in df_train['article']:
    count=len(i.split())
    word_count.append(count)
def avg_list(list):
    return round(sum(list) / len(list))
print('Number of words of the longest article in training dataset:', max(word_count))
print('Number of words of the shortest article in training dataset:', min(word_count))
print('Average number of words of all articles in training dataset:', avg_list(word_count))
ax=sns.displot(word_count, kde=True, bins=15)
plt.xlabel('Number of words in each article')
plt.ylabel('Density')
plt.title('Distribution of words in each article')
plt.show()


#EDA - Word Frequency Analysis of training dataset article
def clean_text(text):
    text=text.lower()
    text=re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text=re.sub("([^\x00-\x7F])+", " ", text)
    return text

df_train['Article_cleaned']=df_train['article'].map(lambda x : clean_text(x))
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
corpus=[word for i in df_train['Article_cleaned'].str.split().values.tolist() for word in i if (word not in stop_words)]
most_common = FreqDist(corpus).most_common(10)
print(most_common)
words, frequency = [],[]
for word, count in most_common:
    words.append(word)
    frequency.append(count)
ax=sns.barplot(x=frequency, y=words)
plt.xlabel('Word appear frequency(M)')
plt.ylabel('Word')
plt.title('Top 10 most appeared words in the articles')
plt.show()

#Applying tokenization to dataset and applying model 't5-small'
model_checkpoint='t5-small'
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)
prefix='summarize:'

def preprocess_function(examples):
    inputs=[prefix+ i for i in examples['article']]
    model_inputs=tokenizer(inputs, max_length=2000, truncation=True)
    labels=tokenizer(text_target=examples['highlights'], max_length=1000, truncation=True)
    model_inputs['labels']=labels['input_ids']
    return model_inputs

# dataset_train=df_train.to_dict('split')
tokenized_train=dataset_train.map(preprocess_function, batched=True)
tokenized_test=dataset_test.map(preprocess_function, batched=True)

data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
batch_size = 5
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
  f"{model_name}-finetuned-CNNv2",
  evaluation_strategy = "epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  weight_decay=0.01,
  save_total_limit=3,
  num_train_epochs=1,
  predict_with_generate=True,
  fp16=True,
  push_to_hub=True,
)

#Defining evaluation method
from datasets import load_metric

rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)

  return {k: round(v, 4) for k, v in result.items()}

#Building the model
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
trainer = Seq2SeqTrainer(
  model,
  args,
  train_dataset=tokenized_train,
  eval_dataset=tokenized_test,
  data_collator=data_collator,
  tokenizer=tokenizer,
  compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub()

#Applying the model to an actual example
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

from transformers import pipeline

summarizer = pipeline("summarization", model="Henry-Chan/t5-small-finetuned-CNNv2")
summarizer(text)

