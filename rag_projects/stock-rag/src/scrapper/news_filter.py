import pandas as pd
import json
from tqdm import tqdm
from dataclasses import dataclass,field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional,Union,Literal
from transformers import Pipeline

from ..base import AgentInput,AgentOutput,BaseAgent

@dataclass
class NewsFilterInput(AgentInput):
    dataset:pd.DataFrame
    date:str
    date_format:str

@dataclass
class NewsFilterOutput(AgentOutput):
    dataset:pd.DataFrame

class StockNewsFilter(BaseAgent):
    def __init__(self,model:Pipeline):
        super().__init__("Stock News Filtering")
        self.model=model

    async def execute(self,input_data:NewsFilterInput) -> NewsFilterOutput:
        input_data.date = pd.to_datetime(input_data.date,format=input_data.date_format)
        dataset = input_data.dataset
        dataset = dataset[dataset['date']>=input_data.date.date()]
        dataset['is_market_news']=None
        dataset['market_confidence']=None
        dataset['sector']=None
        dataset['reason']=None

        for idx,row in  tqdm(dataset.iterrows(),total=len(dataset)):
            prompt = self.__get_prompt(row)
            result = self.model(
                prompt,
            )
            try:
                model_output = json.loads(result[0]['generated_text'][-1]['content'])
                dataset.at[idx, 'is_market_news'] = model_output.get('is_market_news',None)
                dataset.at[idx, 'market_confidence'] = model_output.get('confidence',None)
                dataset.at[idx, 'sector'] = model_output.get('sector',None)
                dataset.at[idx, 'reason'] = model_output.get('reason',None)
            except Exception as e:
                print(result[0]['generated_text'][-1]['content'])
                self.logger.error(f"Failed for {idx} with model output : {model_output} with error {e}")
                
        dataset.dropna(inplace=True)
        dataset = dataset[dataset['is_market_news']==True]

        self.logger.info(f'Total Newses contains market Data : {dataset.shape[0]}')
        return NewsFilterOutput(
            dataset = dataset
            )

    
    def __get_system_prompt(self):
        return """
You are a financial news classifier. Your task is to determine if the following news is related to stock market or not. 

The Stock market news we can seperate in multiple sector : 
    - Automobile and Auto Components Sector
    - Capital Goods Sector
    - Chemicals Sector
    - Construction Materials Sector
    - Construction Sector
    - Consumer Durables Sector
    - Consumer Services Sector
    - Diversified Sector
    - Fast Moving Consumer Goods Sector
    - Financial Services Sector
    - Healthcare Sector
    - Industry Sector
    - Information Technology Sector
    - Media Entertainment & Publication Sector
    - Metals & Mining Sector
    - Oil Gas & Consumable Fuels Sector
    - Power Sector
    - Real Estate Sector
    - Services Sector
    - Telecommunication Sector
    - Textiles Sector

You can also determin if the article is related to any of the followings : 
- Economic Impact Analysis
- Global Government policies or SEBI Regulations
- NSE/BSE Changes

If the news article falls into the above 3 reports then you can consider it as "Global Sector"


Return JSON only.

{
  "is_market_news": true if the news is stock market news, else false,
  "confidence": 0.0,
  "sector": provide any one of the sectors mentioned above you think is most suitable for this article,
  "reason": reason for deciding the sector,
}
"""
    def __get_prompt(self,row):
        prompt = """
Article:

Title:
{title}

Content:
{content}
"""
        message = [
                {'role':'system', 'content' : self.__get_system_prompt()},
                {'role':'user','content':prompt.format(title=row.title,content=row.content)}
            ]
        return message
    