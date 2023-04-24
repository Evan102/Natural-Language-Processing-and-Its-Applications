# Task information
## Context Selection
Train a model to select the relevant context of the question from four contexts.

## Question Answering
Train a model to answer the question from the relevant context.

Note:  
First, do Context Selection task to find the relevant context of the question,and than do Question Answering task to answer the question from the relevant context.

# If you want to train model use:

train_ContextSelection.py  
train_QuestionAnswering.py

Note: 
"train_QuestionAnswering.py" use csv format dataset to train model. 
You can conduct "train_ContextSelection.py" to change json format to csv format.
In "train_ContextSelection.py", you can define your own json format dataset path in "--context_dir"、"--train_dir" and "--valid_dir".

# If you want to do model prdiction use: 

test_ContextSelection.py  
test_QuestionAnswering.py

Note: 
"test_QuestionAnswering.py" use csv format dataset to do prediction. 
You can conduct "test_ContextSelection.py" to change Context Selection prediction json format to csv format.
In "test_ContextSelection.py", you can define your own json format dataset path in "--context_dir" and "--test_dir".

# Bert for Intent Classification and Slot Tagging

train_intent.py
train_slot.py

Note:
In "train_intent.py", you can define your own json format dataset path in "--train_dir"(train.json) 、 "--valid_dir"(eval.json) and "--label2id_dir"(intent2idx.json).

In "train_intent.py", you can define your own json format dataset path in "--train_dir"(train.json) 、 "--valid_dir"(eval.json) and "--label2id_dir"(tag2idx.json). 

# Other files for training and prdiction process:

dataset.py(used in train_ContextSelection.py, test_ContextSelection.py, train_intent.py and train_slot.py)  
utils_qa.py(used in train_run_qa_no_trainer.py and test_run_qa_no_trainer.py)

# Model prediction files for reference
In the directory "reproducefiles"  
CSprediction.csv and CSprediction.json are the prediction from context selection.  
QA_nbest_predictions.json and QAprediction.json are the prediction from question answering.

# PDF file for training result analysis:
report.pdf
