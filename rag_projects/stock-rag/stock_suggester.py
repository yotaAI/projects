import asyncio
from dotenv import load_dotenv
load_dotenv('.env')

import warnings
warnings.filterwarnings("ignore")


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,Pipeline


from src.suggestor import FinanceAgentInput,FinanceAgentOutput,FinanceAgent

async def main():
    model_id = 'Qwen/Qwen3-4B-Instruct-2507'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="bfloat16",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.3,
        max_new_tokens=1024,
        max_length=None,
    )

    finance_input = FinanceAgentInput(
        # company_table_name= 'company_table',
        company_lists_csv = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        stock_table_name= 'stock_news_dataset',
        batch_size=4,
    )
    agent =FinanceAgent(model=pipe)
    finance_out = await agent.execute(finance_input)



if __name__ =='__main__':
    asyncio.run(main())