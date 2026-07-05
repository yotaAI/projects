import pandas as pd
import json
from tqdm import tqdm
from dataclasses import dataclass,field
from typing import Any, Dict, Optional,Union,Literal
from transformers import Pipeline

from ..base import AgentInput,AgentOutput,BaseAgent


@dataclass
class EntityExtractionInput(AgentInput):
    dataset:Optional[pd.DataFrame] = None
    listed_companies:Optional[str] = None
    listed_companies_company_name_col : Optional[str]=None
    batch_size:Optional[int]=4
@dataclass
class EntityExtractionOutput(AgentOutput):
    dataset:Optional[pd.DataFrame] = None
    listed_companies: Optional[list] = None
    company_names: Optional[list] = None
    company_types: Optional[list] = None

class EntityExtraction(BaseAgent):
    def __init__(self,model:Pipeline):
        super().__init__('Entity Extraction')
        self.model=model
    async def execute(self,input_data:EntityExtractionInput):
        self.listed_companies = self.__get_listed_companies(input_data.listed_companies)
        dataset = input_data.dataset
        dataset['company_name']=None
        dataset['company_type']=None
        dataset['company_confidance']=None
        
        prompts = [
            self.__get_prompt(self.listed_companies,input_data.listed_companies_company_name_col,row)
            for _,row in dataset.iterrows()
        ] 
        results = []
        for i in tqdm(range(0, len(prompts), input_data.batch_size), desc="Processing batches"):
            batch = prompts[i:i + input_data.batch_size]
            
            batch_results = self.model(
                batch,
                batch_size=input_data.batch_size
            )

            results.extend(batch_results)

        for idx,model_output in zip(dataset.index,results):
            model_output = json.loads(model_output[0]['generated_text'][-1]['content'])
            try:
                dataset.at[idx,'company_name'] = model_output['name']
                dataset.at[idx,'company_type'] = model_output['type']
                dataset.at[idx,'company_confidance'] = model_output['confidence']
            except Exception as e:
                self.logger.error(f"Failed for {idx} with model output : {model_output} with error {e}")

        dataset = self.__filtering(dataset)

        self.logger.info(f'Final Dataset To Upload : {dataset.shape[0]}')
        return EntityExtractionOutput(
            dataset = dataset,
            listed_companies = self.listed_companies,
            company_names = dataset['company_name'].unique().tolist(),
            company_types = dataset['company_type'].unique().tolist(),
            )
    
    def __get_system_prompt(self):
        system_prompt = """
You are a financial entity extraction system.

Your task is to identify all market-relevant entities mentioned in the article.

Possible entity types:

1. Listed Company
2. NSE/BSE Change
3. SEBI Regulation
4. Government Policy

If None of the above entity type matches, then make name = None and type=None

Rules:

- Match company names only from the provided company list.
- If no company from the list is found, determine whether the article is primarily about:
  - NSE/BSE Change
  - SEBI Regulation
  - Government Policy
- Return ONLY valid JSON.
- Do not add explanations.
- Do not use markdown.

Company List:

{company}

Return format:

{{
    'name' : Listed Companies or 'NSE/BSE Changes' or 'SEBI Regulation' or 'Government Policy',
    'type': 'Listed Company' or 'Regulations' ,
    'confidence' : 0.0,
}}
"""
        return system_prompt

    def __get_prompt(self,listed_companies,company_name_col,row):
        prompt= """
Below sharing the article.
Title:
{title}

Content:
{content}
"""
        message = [
            {
                'role':'system',
                'content' : self.__get_system_prompt().format(company=listed_companies[company_name_col].tolist())
            },
            {
                'role':'user',
                'content': prompt.format(title=row.title,content=row.content)
            }
        ]
        return message
    
    def __get_listed_companies(self,url):
        return pd.read_csv(url)

    def __filtering(self,df):
        df = df.replace(
                    ["None", "nan", ""],
                    None
                )
        df = df[(~df['company_name'].isna()) & (~df['company_type'].isna())]
        return df.reset_index(drop=True)
