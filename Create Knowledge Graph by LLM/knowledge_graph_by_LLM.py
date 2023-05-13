# on 2023/04/29
# The code in this file is modified from:
#   https://apex974.com/articles/explore-langchain-support-for-knowledge-graph
#   You may also refer to this for SPARQL query:
#   https://apex974.com/articles/chatgpt-for-info-retrieval-from-knowledge-graph

# To run this program, create an virtual environment and install necessary packages:
# $ conda create -n langchain python=3.8
# $ conda activate langchain
# $ pip install langchain openai
# $ pip install pyecharts
# Then get you OpenAI api key from: https://platform.openai.com/account/api-keys
# Then run: $ python knowledge_graph_by_LLM.py

""" text-davinci-003 result:
[('吳郭魚', '慈鯛科', '屬於'), 
('吳郭魚', '非洲', '原產於'),
('吳郭魚', '吳、郭兩位先生引進台灣', '由'), 
('吳郭魚', '姓命名', '以'), 
('吳郭魚', '雜食性魚類', '屬於'), 
('吳郭魚', '兇猛', '性情'), 
('吳郭魚', '攻擊性', '具有'), 
('吳郭魚', '是', '成長快')] """

""" gpt-3.5-turbo(ChatGPT) result:
[('吳郭魚', '慈鯛科', '屬於'), 
('吳郭魚', '非洲', '原產於'), 
('吳郭魚', '台灣', '被引進'), 
('吳郭魚', '吳、郭兩位先生的姓氏命名', '以'), 
('吳郭魚', '雜食性魚類', '是'), 
('吳郭魚', '兇猛', '性情'), 
('吳郭魚', '護卵及小魚的習性', '具有'), 
('吳郭魚', '原生種魚類大', '體型比'), 
('吳郭魚', '族群', '迅速建立'), 
('吳郭魚', '原本河川的生態系統', '破壞了')] 

[('吳郭魚', '慈鯛科', '屬於'),
('吳郭魚', '非洲', '原產於'),
('吳郭魚', '吳先生引進台灣', '由'),
('吳郭魚', '郭先生引進 台灣', '由'),
('吳郭魚', '吳郭魚', '名字命名為'),
('吳郭魚', '雜食性魚類', '是'),
('吳郭魚', '兇猛', '性情'),
('吳郭魚', '攻擊性', '具有'),
('吳郭魚', '耐汙染的特性', '具有'),
('吳郭魚', '成長快的特性', '具有'),
('吳郭魚', '繁殖力強的特性', '具有'),
('吳郭魚', '護卵的習性', '具有'),
('吳郭魚', '小魚的習性', '具有'),
('吳郭魚', '比原生種魚類更大的體型', '擁有'),
('吳郭魚', '族群', '迅速建立'),
('吳郭魚', '原本河川的生態系統', '破壞')]"""

import openai
import os
from pyecharts import options as opts
from pyecharts.charts import Graph


os.environ["OPENAI_API_KEY"] = "Your API Key"

# OpenAIChat
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAIChat

from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator

# index_creator = GraphIndexCreator(llm=OpenAI(temperature=0)) # text-davinci-003

index_creator = GraphIndexCreator(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))


# Other text: 'Microsoft Invests $10 Billion in ChatGPT Maker OpenAI.'
separate_text='吳郭魚屬於慈鯛科，原產於非洲。當初由吳、郭兩位先生引進台灣，後來就以他們的姓命名。吳郭魚屬於雜食性魚類，性情兇猛，具攻擊性，耐汙染、成長快、繁殖力強，具有護卵及小魚的習性，加上體型比原生種魚類來得大，因此他們在短暫適應環境後就迅速建立族群，破壞了原本河川的生態系統。'
graph = index_creator.from_text(separate_text)
print(graph.get_triples())

# Separate the result
nodes = []
links = []
for element in graph.get_triples():
    if {'name':element[0]} in nodes:
        if {'name':element[1]} in nodes:
            continue
        else:
            nodes.append({'name':element[1]})
    else:
        nodes.append({'name':element[0]})
        if {'name':element[1]} in nodes:
            continue
        else:
            nodes.append({'name':element[1]})
            
    links.append({'source': element[0], 'target': element[1], 'value': element[2]})
    
# Create the chart
graph = Graph()
graph.add("", nodes, links, repulsion=8000)

# Set the global options for the chart
graph.set_global_opts(title_opts=opts.TitleOpts(title="Knowledge Graph"))

# Render the chart
graph.render("knowledge_graph.html")
