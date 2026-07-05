import os,sys
import pandas as pd
import psycopg
import uuid
from dataclasses import dataclass,field
from typing import Any, Dict, Optional,Union,Literal
from ..base import AgentInput,AgentOutput,BaseAgent

@dataclass
class UploadingInput(AgentInput):
    dataset:pd.DataFrame
    uploading_columns:list
    sql_column_names:list
    table_name:str
    conflict_col:Optional[str]=None

class UploadToPLSQL(BaseAgent):
    def __init__(self):
        super().__init__('Uploading to PLSQL')
        self.__connect()
    async def execute(self, input_data:UploadingInput):
       
       self.__uploading_rows(input_data) 
       
    def __connect(self):
        self.conn = psycopg.connect(
                host=os.getenv('PS_HOST'),
                port=os.getenv('PS_PORT'),
                dbname=os.getenv('PS_DBNAME'),
                user=os.getenv('PS_USER'),
                password=os.getenv('PS_PASSWORD'),
                )
        self.sql_cursor = self.conn.cursor()

        self.logger.info("PSQL Connected Successfully!")

    def __uploading_rows(self,input_data:UploadingInput):
        records = [
            (
                str(uuid.uuid4()),
                *[row[col] for col in input_data.uploading_columns]
            )
            for _,row in input_data.dataset.iterrows()
        ]
        column_names = ', '.join(input_data.sql_column_names)
        placeholders = ', '.join(['%s'] * len(input_data.sql_column_names))
        comment = f"""
INSERT INTO {input_data.table_name} (
{column_names}
)
VALUES ({placeholders})
"""
        if input_data.conflict_col!=None:
            comment+=f"ON CONFLICT ({input_data.conflict_col}) DO NOTHING"

        
        self.sql_cursor.executemany(
            comment,
        records,
            )
        self.conn.commit()
        
        self.logger.info("All records are uploaded to SQL!")
    def _truncate_table(self,table_name):
        self.sql_cursor.execute(f"DELETE FROM {table_name};")
        self.logger.info("Table is truncated!")