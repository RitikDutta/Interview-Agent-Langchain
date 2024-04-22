import pathlib
import textwrap

import google.generativeai as genai

from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


class DataScienceInterviewAssistant:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.chat = self.model.start_chat(history=[])

    def get_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def init_chat(self):
        while(1):
            message = input("Message: ")
            if message=='exit': break
            response = self.chat.send_message(message)
            print(response.text)
            self.get_history()

    def get_history(self):
        print(self.chat.history)

if __name__ == "__main__":
    ds = DataScienceInterviewAssistant()
    ds.get_models()
    ds.init_chat()
    ds.get_history()