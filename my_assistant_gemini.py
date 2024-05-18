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
genai.configure(api_key=GOOGLE_API_KEY)


class DataScienceInterviewAssistant:
    def __init__(self, instruction, current_user):
        self.current_user = current_user
        self.usermanager = UserManager()
        genai.configure(api_key=GOOGLE_API_KEY)
        instructions = self.get_instructions(instructions=instruction)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=instructions)
        chat = self.usermanager.get_chat(self.current_user)
        if chat:
            chat = self.reconstruct_format(chat)
        else:
            chat=[]
        self.chat = self.model.start_chat(history=chat)
        self.msg = ''

    def get_instructions(self, instructions):
        with open(instructions, 'r') as file:
            instructions = file.read()
        return instructions

    def get_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def conduct_interview(self, question):
        response = self.chat.send_message(question)
        # print("JSON", self.convert_json_string_to_dict(response.text))
        chat = self.convert_format(self.get_history())
        self.get_history()
        self.usermanager.add_or_update_chat(self.current_user, chat)
        return "hioiiiiiiasdiaid", 9

    def get_history(self):
        return self.chat.history

    def get_thread_id(self):
        return '123456789'
    
    def get_messages(self, custom_name):
        chat = self.usermanager.get_chat(self.current_user)
        if chat:
            chat = chat[::-1]  # Reverse the chat list
            if custom_name:
            #modify the chat list with replaced text
                for index in range(len(chat)):
                    # Replace "User" with "xyz" and update the list at the same position
                    chat[index] = chat[index].replace("User", custom_name)
                    # print("MESSAGE: ", chat[index])  # Print the updated message

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
                        pass
                part = {"text": text}
            else:
                continue
            
            entry = {
                "parts": [part],
                "role": role
            }
            data.append(entry)
        return data

    def convert_json_string_to_dict(self, json_string):
        # We only need to remove from start and end not to replace from the main message
        start_delimiter = "```json"
        end_delimiter = "```"
        start_index = json_string.find(start_delimiter) + len(start_delimiter)
        end_index = json_string.rfind(end_delimiter)
        
        if start_index > -1 and end_index > -1:
            # Extract the JSON content from between the delimiters
            json_string = json_string[start_index:end_index].strip()

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return None


    def upload_file_to_gemini(self, path, mime_type=None):
        """uploads file to Gemini."""
        try:
            file = genai.upload_file(path, mime_type=mime_type)
            log = f"Uploaded file '{file.display_name}' as: {file.uri}"
            print(log)
            return file, log
        except Exception as e:
            log = f"Failed to upload file to Gemini: {str(e)}"
            print(log)
            return None, log



