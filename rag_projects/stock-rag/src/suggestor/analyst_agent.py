from dataclasses import dataclass
from typing import Any, Dict, Optional,Union,Literal
import json
from tqdm import tqdm
import pandas as pd
# Pushing data to PostGrad SQL
import psycopg
import os 
from transformers import Pipeline


from ..base import BaseAgent, AgentInput, AgentOutput
from ..scrapper import UploadingInput, UploadToPLSQL
from .prompts import SYSTEM_PROMPT, PROMPT

@dataclass
class AnalystAgentInput(AgentInput):
    # company_table_name:str
    company_df:pd.DataFrame
    stock_table_name:str
    batch_size:int = 8

@dataclass
class AnalystAgentOutput(AgentOutput):
    market_news:Dict

class AnalystAgent(BaseAgent):
    def __init__(self,model:Pipeline):
        super().__init__("Analyst Agent")
        self.model=model
        
        self.conn = psycopg.connect(
                                        host=os.getenv("PS_HOST"),
                                        port=os.getenv("PS_PORT"),
                                        dbname=os.getenv("PS_DBNAME"),
                                        user=os.getenv("PS_USER"),
                                        password=os.getenv("PS_PASSWORD"),
                                    )
        
        self.company_query = """
                        SELECT *
                        FROM {company_table};
                        """
        self.news_query = """
                    SELECT *
                    FROM {stock_table}
                    WHERE SECTOR LIKE '{industry}%'
                    """
        
        self.market_news = dict()
    
    async def execute(self,input_data:AnalystAgentInput) ->AnalystAgentOutput:
        
        self.logger.info(f'Company Table loaded with {input_data.company_df.shape[0]} data points.')

        self.logger.info('Starting Analysis...')

        for idx,row in  input_data.company_df.iterrows():
            company = row['Company Name']
            industry = row['Industry']
            self.market_news[company] = {
                'company' : company,
                'industry': industry,
                'score' : 0.0,
                'sentiments': [],
            }
            news_df = pd.read_sql_query(
                self.news_query.format(
                    stock_table=input_data.stock_table_name,
                    industry=industry
                    ), self.conn
                )

            for idx,row in tqdm(news_df.iterrows(),total=len(news_df), desc=f"Company :'{company}' News article : "):
                message = [
                    {
                        'role':'system',
                        'content' : SYSTEM_PROMPT,
                    },
                    {
                        'role':'user',
                        'content': PROMPT.format(company_name = company,
                                        industry = industry, 
                                        title = row.title, 
                                        content = row.content),
                    },
                ]

                result = self.model(message)
                sentiment = json.loads(result[0]['generated_text'][-1]['content'])
                if sentiment['sentiment'].lower()!='neutral':
                    self.market_news[company]['sentiments'].append(sentiment)
                
            self.market_news[company]['score'] = 0 if len(self.market_news[company]['sentiments'])==0 else sum([sentiment['sentiment_score'] * sentiment['confidence'] for sentiment in self.market_news[company]['sentiments']]) / len(self.market_news[company]['sentiments'])


        return AnalystAgentOutput(
            market_news = self.market_news,
        )
