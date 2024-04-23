import pathlib
import textwrap
import random
import google.generativeai as genai
import re
from database.user_manager import UserManager

from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


class DataScienceInterviewAssistant:
    def __init__(self, instruction, current_user):
        self.current_user = current_user
        self.usermanager = UserManager()
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.chat = self.model.start_chat(history=[])
        self.msg = ''

    def get_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def conduct_interview(self, question):
        response = self.chat.send_message(question)
        print(response.text)
        self.msg = response.text

        chat = self.convert_format(self.get_history())
        self.get_history()
        print("current user = ", self.current_user)
        self.usermanager.add_or_update_chat(self.current_user, chat)
        return "hioiiiiiiasdiaid", 9

    def get_history(self):
        print(self.chat.history)
        return self.chat.history

    def get_thread_id(self):
        return '123456789'
    
    def get_messages(self, custom_name):
        # hard coding
        chat = self.usermanager.get_chat(self.current_user)
        
        return chat 


         
    def convert_format(self, data):
        result = []
        # Iterate over the data in reverse order to prioritize assistant messages
        for entry in reversed(data):
            role_prefix = "User:" if entry.role == "user" else "Assistant:"
            # Directly access the first part assuming there's exactly one part per entry
            if entry.parts:  # Check if parts are not empty
                part = entry.parts[0]
                text = part.text.split('.')[0] + '.' if '.' in part.text else part.text
                result.append(f"{role_prefix} {text}")
        return result



    


if __name__ == "__main__":
    ds = DataScienceInterviewAssistant()
    ds.get_models()
    ds.init_chat()
    ds.get_history()