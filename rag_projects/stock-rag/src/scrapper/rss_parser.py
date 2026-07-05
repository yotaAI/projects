import pandas as pd
from dataclasses import dataclass,field
from typing import Any, Dict, Optional,Union,Literal
import requests
from bs4 import BeautifulSoup

from ..base import AgentInput,AgentOutput,BaseAgent

@dataclass
class RSSParsingInput(AgentInput):
    link:str | list[str]
    date_format:str
    conf:dict = field(
        default_factory=lambda:{
    })

@dataclass
class RSSParsingOutput(AgentOutput):
    dataset:pd.DataFrame
    columns:tuple

class RSSParser(BaseAgent):
    def __init__(self):
        super().__init__('RSSParser')
    
    async def execute(self, input_data:RSSParsingInput):

        if isinstance(input_data.link,str):
            articles = await self.__fetch_data(input_data.link,input_data.conf)
        elif isinstance(input_data.link,(tuple,list,set)):
            articles = pd.concat(
                [
                    await self.__fetch_data(link,input_data.conf) for link in input_data.link
                ]
            )
        else:
            raise ValueError("Accept ony string,tuple,list,set.")
        
        articles['date'] = pd.to_datetime(
            articles['date'],
            format = input_data.date_format,
        ).dt.date

        self.logger.info(f'Total Data Fetched From XML(s) : {articles.shape[0]}')
        return RSSParsingOutput(
                    dataset=articles,
                    columns=articles.columns
        )
    
    async def __fetch_data(self,link,conf):
        xml = requests.get(link).text
        rss = BeautifulSoup(xml,'xml')
        
        newses = []

        news_items = rss.find_all(conf['item_tag'])
        for news in news_items:
            id = news.find(conf['id_tag']).text.strip()
            title = news.find(conf['title_tag']).text.strip()
            description = news.find(conf['description_tag']).text.strip()
            content = news.find(conf['content_tag']).text.strip()
            article = BeautifulSoup(content,'html.parser').get_text(' ',strip=True)

            date = news.find(conf['date_tag']).text.strip()

            newses.append({
                'id':id,
                'title':title,
                'description':description,
                'content':article,
                'date':date,
            })
        newses = pd.DataFrame(newses)
        return newses