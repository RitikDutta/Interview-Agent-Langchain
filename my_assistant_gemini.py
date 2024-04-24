import pathlib
import textwrap
import random
import google.generativeai as genai
import re
from database.user_manager import UserManager
import json

from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


class DataScienceInterviewAssistant:
    def __init__(self, instruction, current_user):
        self.current_user = current_user
        self.usermanager = UserManager()
        genai.configure(api_key=GOOGLE_API_KEY)
        instructions = self.get_instructions()
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=instructions)
        chat = self.usermanager.get_chat(self.current_user)
        if chat:
            chat = self.reconstruct_format(chat)
        else:
            chat=[]
        self.chat = self.model.start_chat(history=chat)
        self.msg = ''

    def get_instructions(self):
        with open('instructions.txt', 'r') as file:
            instructions = file.read()
        return instructions

    def get_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def conduct_interview(self, question):
        response = self.chat.send_message(question)
        print(response.text)
        self.msg = response.text
        print("RESPONSE: ", response.text)
        print("before conversion: ", self.get_history())
        chat = self.convert_format(self.get_history())
        print("after conversion: ", chat)
        self.get_history()
        print("current user = ", self.current_user)
        self.usermanager.add_or_update_chat(self.current_user, chat)
        print("RRRRRRRRRRRRRRRRRRRRRRRRR", self.reconstruct_format(chat))
        return "hioiiiiiiasdiaid", 9

    def get_history(self):
        print(self.chat.history)
        return self.chat.history

    def get_thread_id(self):
        return '123456789'
    
    def get_messages(self, custom_name):
        # hard coding
        chat = self.usermanager.get_chat(self.current_user)
        if chat:
            chat = chat[::-1]
        print("CHATTTTT: ", chat)
        
        return chat 


         
    def convert_format(self, data):
        result = []
        for entry in data:
            # Access attributes of Content object directly if it's not a dictionary
            role_prefix = "User:" if entry.role == "user" else "Assistant:"
            for part in entry.parts:
                # Assuming 'text' can be accessed directly as an attribute of objects in parts
                text = part.text
                result.append(f"{role_prefix} {text}")
        return result

    def reconstruct_format(self, chat_list):
        data = []
        for message in chat_list:
            if message.startswith('User:'):
                role = 'user'
                text = message[len('User: '):]
                part = {"text": text}
            elif message.startswith('Assistant:'):
                role = 'model'
                text = message[len('Assistant: '):]
                # Special handling for JSON content, assuming it starts with '`json'
                if text.strip().startswith('```json'):
                    try:
                        # Attempt to extract the JSON part correctly
                        json_str = text.strip()[6:-4]  # Remove the ```json and ending ```
                        json_obj = json.loads(json_str)
                        text = f"```json\n{json.dumps(json_obj, indent=2)}\n``` \n"
                    except json.JSONDecodeError:
                        print("Error decoding JSON from message.")
                part = {"text": text}
            else:
                continue
            
            entry = {
                "parts": [part],
                "role": role
            }
            data.append(entry)
        return data


    


if __name__ == "__main__":
    ds = DataScienceInterviewAssistant()
    ds.get_models()
    ds.init_chat()
    ds.get_history()