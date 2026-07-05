from dataclasses import dataclass
from typing import Any, Dict, Optional,Union,Literal
import pandas as pd
# Pushing data to PostGrad SQL
from datetime import date
from transformers import Pipeline


from ..base import BaseAgent, AgentInput, AgentOutput
from ..scrapper import UploadingInput, UploadToPLSQL
from .analyst_agent import AnalystAgent,AnalystAgentInput
from .suggester_agent import SuggesterAgent,SuggesterAgentInput

@dataclass
class FinanceAgentInput(AgentInput):
    # company_table_name:str
    company_lists_csv:str
    stock_table_name:str
    batch_size:int = 8
@dataclass
class FinanceAgentOutput(AgentOutput):
    stocks:Dict

class FinanceAgent(BaseAgent):
    def __init__(self,model:Pipeline):
        super().__init__("Finance Agent")
        self.model=model
        self.analyst_agent = AnalystAgent(self.model)
        self.suggestion_agent = SuggesterAgent(self.model)
        self.psql_uploader = UploadToPLSQL()
    
    async def execute(self,input_data:FinanceAgentInput) ->FinanceAgentOutput:
        company_df = pd.read_csv(input_data.company_lists_csv)
        analyst_input = AnalystAgentInput(
            # company_table_name=input_data.company_table_name,
            company_df = company_df,
            stock_table_name=input_data.stock_table_name,
            batch_size=input_data.batch_size,
        )
        analyst_out = await self.analyst_agent.execute(analyst_input)

        suggestion_input = SuggesterAgentInput(
            market_news=analyst_out.market_news,
            batch_size=input_data.batch_size,
        )

        suggegstion_output = await self.suggestion_agent.execute(suggestion_input)

        investment_sugggestions = pd.DataFrame(
            [[key,value['Investment'],value['Time Period'],value['Expected Profit']] for key,value in suggegstion_output.suggestions.items()],
            columns = ['Industry','Investment','Time Period','Expected Profit']
        )
        investment_sugggestions['Date'] = date.today()

        self.logger.info('Uploading to PSQL...')

        psql_input = UploadingInput(
                                dataset = investment_sugggestions,
                                uploading_columns = ['Industry','Investment',	'Time Period',	'Expected Profit',	'Date'],
                                sql_column_names = ['id', 'company','investment','time_period','expected_profit','created_at'],
                                table_name= 'stock_suggestions',
                            )
        
        self.psql_uploader._truncate_table(psql_input.table_name)
        await self.psql_uploader.execute(psql_input)

        return FinanceAgentOutput(
            stocks = investment_sugggestions,
        )
