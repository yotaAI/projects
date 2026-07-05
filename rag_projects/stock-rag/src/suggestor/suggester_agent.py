from dataclasses import dataclass
from typing import Any, Dict, Optional,Union,Literal
import json
# Pushing data to PostGrad SQL
from transformers import Pipeline
from tqdm import tqdm

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

        companies = []
        messages = []

        for company,reasoning in tqdm(market_news.items(),total=len(market_news.keys())):
            industry = reasoning['industry']

            positive_factors = [
                factor
                for sentiment in reasoning["sentiments"]
                for factor in sentiment["key_positive_factors"]
            ]

            negative_factors = [
                factor
                for sentiment in reasoning["sentiments"]
                for factor in sentiment["key_negative_factors"]
            ]

            reasonings = [
                sentiment["reasoning"]
                for sentiment in reasoning["sentiments"]
                if sentiment["sentiment"].lower() != "neutral"
            ]

            positive_factors = (
                "  - " + "\n  - ".join(positive_factors)
                if positive_factors else "None"
            )

            negative_factors = (
                "  - " + "\n  - ".join(negative_factors)
                if negative_factors else "None"
            )

            reasonings = (
                "  - " + "\n  - ".join(reasonings)
                if reasonings else "None"
            )

            message = [
                {
                    "role": "system",
                    "content": SUGGESTER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": SUGGESTER_PROMPT.format(
                        company=company,
                        industry=industry,
                        positive_factors=positive_factors,
                        negagtive_factors=negative_factors,
                        reasonings=reasonings,
                    ),
                },
            ]
            companies.append(company)
            messages.append(message)

        # Batched inference
        results = self.model(
            messages,
            batch_size=input_data.batch_size,
            truncation=True,
        )

        # Parse outputs
        for company, result in zip(companies, results):
            output = json.loads(result[0]["generated_text"][-1]["content"])
            self.company_suggestions[company] = output

        return SuggesterAgentOutput(
            suggestions = self.company_suggestions
            )