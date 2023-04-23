# Task information

## Intent Classification: 

Input: Text  
"i dont like my current insurance plan and want a new one",  
"when will my american express credit card expire",  
"how would i get to city hall via bus"

Output: Intent  
"insurance_change",  
"expiration_date",  
"directions"

## Slot Tagging
Input: Text  
"A table today for myself and 3 others",  
"My three children and i are in the party"

Output: Intent  
"O O B-date O B-people I-people I-people O",  
"B-people I-people I-people I-people I-people O O O O"


# If you want to train model use: 
train_intent.py  
train_slot.py


# If you want to do model prdiction use: 

test_intent.py  
test_slot.py

# model files for LSTM & GRU: 

intentmodel.py  
slotmodel.py

# other files for training and prdiction process:

dataset.py  
utils.py

# pdf report:

report.pdf