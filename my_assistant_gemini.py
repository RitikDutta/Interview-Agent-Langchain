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
        print("THIS IS STORING ", chat)
        print("THIS IS STORING ", type(chat))
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
    
    # regex to remove character that affect the string to convert to json
    def clean_selected_text(self, input_text):
        # remove newlines from the entire string
        input_text = input_text.replace('\n', ' ')
        
        # egex to match the text between single or double quotes in the values
        regex = r": '(.*?)',|: '(.*?)' } |: \"(.*?)\",|: \"(.*?)\" }"
        
        # function to remove special characters from matched groups
        def clean_match(match):
            return re.sub(r"[:\'\"{}]", "", match)
        
        # function to replace matches in the original string
        def replace_matches(match):
            groups = match.groups()
            for group in groups:
                if group:
                    cleaned = clean_match(group)
                    return match.group(0).replace(group, cleaned)
            return match.group(0)
        
        # substitute matches with cleaned text
        cleaned_text = re.sub(regex, replace_matches, input_text)
        
        return cleaned_text




    def convert_json_string_to_dict(self, json_string):
        # We only need to remove from start and end not to replace from the main message
        clean_json = self.clean_selected_text(json_string)
        print(clean_json)
        start_delimiter = "```json"
        end_delimiter = "```"
        start_index = clean_json.find(start_delimiter) + len(start_delimiter)
        end_index = clean_json.rfind(end_delimiter)
        
        if start_index > -1 and end_index > -1:
            # Extract the JSON content from between the delimiters
            clean_json = clean_json[start_index:end_index].strip()

        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return None


    # # this function is used when : is present in json string which cause it to give error
    # def remove_unwanted_colons(self, json_string):

    #     feedback_pattern = r'("feedback":)'
    #     score_pattern = r'("score":)'
    #     next_question_pattern = r'("next_question":)'

    #     json_string = re.sub(feedback_pattern, r'\1PLACEHOLDER_FEEDBACK', json_string)
    #     json_string = re.sub(score_pattern, r'\1PLACEHOLDER_SCORE', json_string)
    #     json_string = re.sub(next_question_pattern, r'\1PLACEHOLDER_NEXT_QUESTION', json_string)

    #     json_string = json_string.replace(":", "")
    #     json_string = json_string.replace("\n", "")

    #     json_string = json_string.replace('PLACEHOLDER_FEEDBACK', ':')
    #     json_string = json_string.replace('PLACEHOLDER_SCORE', ':')
    #     json_string = json_string.replace('PLACEHOLDER_NEXT_QUESTION', ':')

    #     return json_string

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

    def init_chat(self, language):
        # check if chat is empty
        chat = self.usermanager.get_chat(self.current_user)
        if not chat:
            if language == "English":
                init_chat = ['User: test', 'Assistant: ```json\n{\n  "feedback": "",\n  "score": "",\n  "next_question": "Hello! I\'m your Mock Interview Mentor. Which domain are you preparing for?"\n}\n```\n']
            elif language == "Hindi":
                init_chat = ['User: test', 'Assistant: ```json\n{\n  "feedback": "",\n  "score": "",\n  "next_question": "नमस्ते! मैं आपका मॉक इंटरव्यू मेंटर हूँ। आप किस क्षेत्र की तैयारी कर रहे हैं?"\n}\n```\n']
            self.usermanager.add_or_update_chat(self.current_user, init_chat)

    # def reset_chat(self):
    #     init_chat = ['User: test', 'Assistant: ```json\n{\n  "feedback": "",\n  "score": "",\n  "next_question": "Hello! I\'m your Mock Interview Mentor. Which domain are you preparing for?"\n}\n```\n']
    #     print("THIS IS ME STORING ", type(init_chat))
    #     print("CHAT INIT FROM gemini")

    #     self.usermanager.add_or_update_chat(self.current_user, init_chat)


