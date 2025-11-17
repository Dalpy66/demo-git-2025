import abc
from google.genai.types import GenerateContentConfig
from dataclasses import dataclass
from enum import Enum
from openaiclass import OpenAIClient
from genaiclass import GoogleGenAIClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class GenerativeAIClient(abc.ABC):

  @abc.abstractmethod
  def generate(self, prompt: str, **kwargs) -> str:
        ...

class Provider(Enum):
    OPENAI = 'openai'
    GOOGLE = 'google'

class GenerativeAIClientFactory:
    
    @staticmethod
    def create_client(provider: Provider) -> GenerativeAIClient:
        if provider == Provider.OPENAI:
            return OpenAIClient()
        elif provider == Provider.GOOGLE:
            return GoogleGenAIClient()
        else:
            raise ValueError(f"Unknown provider: {provider}")