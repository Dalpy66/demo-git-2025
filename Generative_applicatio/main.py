from openaiclass import OpenAIClient
from genaiclass import GoogleGenAIClient
from generativeFactory import GenerativeAIClientFactory, Provider


def main():
    client = OpenAIClient()
    response = client.generate("Hello, how are you?")
    print(response)

    client = GoogleGenAIClient()
    response = client.generate("Hello, how are you?")
    print(response)

    client = GenerativeAIClientFactory.create_client(Provider.OPENAI)
    response = client.generate("Hello, how are you?")
    print(response)

if __name__ == "__main__":
    main()