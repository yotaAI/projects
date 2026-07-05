from dataclasses import dataclass
from typing import Any, Dict, Optional,Union,Literal
import json
# Pushing data to PostGrad SQL
from transformers import Pipeline


from ..base import BaseAgent, AgentInput, AgentOutput
from .prompts import SUGGESTER_SYSTEM_PROMPT, SUGGESTER_PROMPT

@dataclass
class SuggesterAgentInput(AgentInput):
    market_news:Dict
    batch_size:int = 8
@dataclass
class SuggesterAgentOutput(AgentOutput):
    suggestions:Dict

class SuggesterAgent(BaseAgent):
    def __init__(self,model:Pipeline):
        super().__init__("Finance Agent")
        self.model=model
        self.company_suggestions = dict()
    
    async def execute(self,input_data:SuggesterAgentInput) ->SuggesterAgentOutput:
        market_news = input_data.market_news
        for company,resoning in market_news.items():
            industry = resoning['industry']
            positive_factors = [ factor for sentiment in resoning['sentiments'] for factor in sentiment['key_positive_factors']]
            negagtive_factors = [ factor for sentiment in resoning['sentiments'] for factor in sentiment['key_negative_factors']]
            reasonings = [
                sentiment["reasoning"]
                for sentiment in resoning["sentiments"]
                if sentiment["sentiment"].lower() != "neutral"
            ]

            positive_factors = '  - ' +"\n  - ".join(positive_factors)
            negagtive_factors = '  - ' +"\n  - ".join(negagtive_factors)
            reasonings = '  - ' +"\n  - ".join(reasonings)
        
            message  = [
                        {
                            'role':'system',
                            'content':SUGGESTER_SYSTEM_PROMPT,
                        },
                        {
                            'role': 'user',
                            'content': SUGGESTER_PROMPT.format(
                                company=company,
                                industry = industry,
                                positive_factors = positive_factors,
                                negagtive_factors = negagtive_factors,
                                reasonings = reasonings,
                            ),
                        }
                    ]
            result = self.model(message)
            output = json.loads(result[0]['generated_text'][-1]['content'])

            self.company_suggestions[company] = output

        return SuggesterAgentOutput(
            suggestions = self.company_suggestions
            )