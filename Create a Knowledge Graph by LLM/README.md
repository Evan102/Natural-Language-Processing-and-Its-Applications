# Create a Knowledge Graph by LLM

## Run knowledge_graph_by_LLM.py to get a knowledge_graph.html created by large language model;
## Remember to set your OpenAI API key in knowledge_graph_by_LLM.py: 
python knowledge_graph_by_LLM.py

## My example:
separate_text=
Solar energy plays a critical role in achieving net-zero emissions goals. Accurately predicting long-term solar power potential is essential for identifying optimal locations for solar power deployment. 
This is however a challenging task since the significant uncertainty resulting from varying weather conditions and geographical locations on solar power output. 
There is therefore an urgent need to develop reliable prediction models to facilitate successful solar power deployment. This paper compares the performance of three artificial intelligence models - XGBoost, MLP, and LSTM - in predicting the daily solar power capacity factor in Taiwan.

## Triple table result:
[（entity1, relation, entity2）]:  
[('Solar energy', 'plays a critical role in', 'achieving net-zero emissions goals'),  
('predicting', 'solar power potential', 'is essential'),  
('identifying', 'optimal locations', 'for solar power deployment'),  
('significant uncertainty', 'results from', 'varying weather conditions and geographical locations'),  
('developing', 'reliable prediction models', 'is urgent'),  
('this paper', 'compares the performance of', 'three artificial intelligence models'),   
('XGBoost', 'predicts', 'daily solar power capacity factor in Taiwan'),  
('MLP', 'predicts', 'daily solar power capacity factor in Taiwan'),  
('LSTM', 'predicts', 'daily solar power capacity factor in Taiwan')]

## Knowledge graph:
<img src="https://github.com/Evan102/Natural-Language-Processing-and-Its-Applications/blob/main/Create%20a%20Knowledge%20Graph%20by%20LLM/solar%20energy%20knowledge%20graph.png"  width="100%" height="60%">
