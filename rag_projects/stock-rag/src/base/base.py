from dataclasses import dataclass,field
from abc import ABC, abstractmethod
import logging
import warnings
warnings.filterwarnings("ignore")

@dataclass
class AgentInput:
    pass

@dataclass
class AgentOutput:
    pass
class BaseAgent(ABC):
    def __init__(self,name:str):
        self.name = name
        self.history = []
        self.__set_loggger(name)

    def __set_loggger(self,name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)
        
    @abstractmethod
    async def execute(self, input_data: AgentInput) -> AgentOutput:
        pass
