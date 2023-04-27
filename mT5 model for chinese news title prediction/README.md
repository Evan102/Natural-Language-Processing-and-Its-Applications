# mT5 model for chinese news title prediction
## input: news content  
從小就很會念書的李悅寧， 在眾人殷殷期盼下，以榜首之姿進入臺大醫學院， 但始終忘不了對天文的熱情。大學四年級一場遠行後，她決心遠赴法國攻讀天文博士。 從小沒想過當老師的她，再度跌破眾人眼鏡返台任教，

## output: news title
榜首進台大醫科卻休學 、27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教

# Download dataset link:
https://drive.google.com/file/d/1GRRD9vjy0JxddqO6--EbfPHx71MB4dHQ/view?usp=sharing

# If you want to train model use:

run_summarization_no_trainer.py

Note: 
In "run_summarization_no_trainer.py", you can define your own jsonl format dataset path in "--train_file" and "--validation_file".

# If you want to do model prdiction use: 

test_run_summarization_no_trainer.py

Note: 
In "test_run_summarization_no_trainer.py", you can define your own input jsonl format dataset path in "--validation_file" and output jsonl format dataset path in "--bestpredictjsonl_dir".

# Other files for evaluting model performance:

tw_rouge  
eval.py

Note:
"tw_rouge" directory for validation dataset scoring when runnung "run_summarization_no_trainer.py".
"python eval.py -r public.jsonl -s submission.jsonl" to score model prediction.

# PDF file for training result analysis:

report.pdf

