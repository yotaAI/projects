from dataclasses import dataclass,field
from typing import Any, Dict, Optional,Union,Literal
import os
from transformers import Pipeline

from ..base import AgentInput,AgentOutput,BaseAgent
from .rss_parser import RSSParsingInput,RSSParser
from .news_filter import NewsFilterInput,StockNewsFilter
from .entity_extractor import EntityExtractionInput,EntityExtraction
from .plsql_uploader import UploadingInput, UploadToPLSQL


@dataclass
class NewsFetcherInput(AgentInput):
    news_link:str | list[str]
    scrapper_date_format:str
    filter_date:str
    sql_uploading_columns:list
    sql_column_names:list
    conflict_col:str
    table_name:str
    scrapper_config:Dict

class NewsFetcher(BaseAgent):
    def __init__(self,model:Pipeline,parser:Literal['rss','money_control']):
        super().__init__("NewsFetcher")
        self.model=model
        if parser=='rss':
            self.scrapper = RSSParser()
        self.news_filter = StockNewsFilter(self.model)
        self.psql_uploader = UploadToPLSQL()
        
        self.logger.info('Initiated!')

    async def execute(self, input_data:NewsFetcherInput):
        self.logger.info('Scrapping Newses..')
        scrapper_input = RSSParsingInput(
            link=input_data.news_link,
            conf = input_data.scrapper_config,
            date_format=input_data.scrapper_date_format,
        )
        scrapper_out = await self.scrapper.execute(scrapper_input)

        self.logger.info('Filtering Newses for Stock Market News.')
        
        news_filter_input = NewsFilterInput(
            dataset=scrapper_out.dataset,
            date=input_data.filter_date,
            date_format='%d-%m-%Y',
        )
        news_filter_out = await self.news_filter.execute(news_filter_input)

        self.logger.info('Uploading Data to SQL')
        psql_input = UploadingInput(
                        dataset=news_filter_out.dataset,
                        uploading_columns = input_data.sql_uploading_columns,
                        sql_column_names=input_data.sql_column_names,
                        conflict_col=input_data.conflict_col,
                        table_name=input_data.table_name,
                    )
        await self.psql_uploader.execute(psql_input)

        self.logger.info('Process Complete!')
