import asyncio
from dotenv import load_dotenv
load_dotenv('.env')

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Loading Local Module
from src import NewsFetcherInput,NewsFetcher


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

    news_scrapper_input = NewsFetcherInput(
        news_link = [
            'https://www.etnownews.com/feeds/gns-etn-companies.xml'
            'https://www.etnownews.com/feeds/gns-etn-latest.xml',
            'https://www.etnownews.com/feeds/gns-etn-news.xml',
            'https://www.etnownews.com/feeds/gns-etn-markets.xml',
            'https://www.etnownews.com/feeds/gns-etn-companies.xml',
            'https://www.etnownews.com/feeds/gns-etn-personal-finance.xml',
            'https://www.etnownews.com/feeds/gns-etn-cryptocurrency.xml',
            'https://www.etnownews.com/feeds/gns-etn-news-letter.xml',
            'https://www.etnownews.com/feeds/gns-etn-technology.xml',
            'https://www.etnownews.com/feeds/gns-etn-economy.xml',
            'https://www.etnownews.com/feeds/gns-etn-real-estate.xml',
            'https://www.etnownews.com/feeds/gns-etn-budget.xml',
            'https://www.etnownews.com/feeds/gns-etn-success-stories.xml',
            ],
        scrapper_config={
            'item_tag':'item',
            'id_tag' : 'guid',
            'title_tag': 'title',
            'description_tag':'description',
            'content_tag':'content:encoded',
            'date_tag':'pubDate',
        },
        scrapper_date_format="%a, %d %b %Y %H:%M:%S %z",
        filter_date = '01-04-2025',
        sql_uploading_columns= ['id',"title","content","date","is_market_news","sector","market_confidence"],
        sql_column_names=['id','news_id','title','content','published_at','is_market_news','sector','market_confidence_score'],
        conflict_col='news_id',
        table_name='stock_news_dataset',
    )

    news_scrapper = NewsFetcher(model=pipe,parser='rss')

    await news_scrapper.execute(news_scrapper_input)

if __name__ =='__main__':
    asyncio.run(main())