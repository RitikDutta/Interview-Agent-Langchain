from database.firestoreCRUD import FirestoreCRUD
from firebase_admin import credentials, firestore
import firebase_admin

class UserManager:
    def __init__(self):
        self.firestore_crud = FirestoreCRUD
        self.credential_path = 'database/interview-mentor-firebase-adminsdk-sguq7-aee5c6cca8.json'
        self.collection_name = 'users'
        if not firebase_admin._apps:
            cred = credentials.Certificate(self.credential_path)
            firebase_admin.initialize_app(cred)

        self.firestore_crud = FirestoreCRUD(self.credential_path, self.collection_name)
    def initialize_user(self, user_id, name, email):
        user_data = self.firestore_crud.read_document(user_id)
        if not user_data:
            # User does not exist, so create new user
            details = {'name': name, 'email': email}
            performance = {}  # Initialize performance data, if any
            chat = {}
            
            user_data = {'details': details, 'performance': performance, 'chat': chat}
            self.firestore_crud.create_document(user_id, user_data)


    def add_thread_id(self, user_id, thread_id):
        # Add thread_id to the user's performance data
        user_data = self.firestore_crud.read_document(user_id)
        if user_data:
            user_data['performance']['thread_id'] = thread_id
            self.firestore_crud.update_document(user_id, user_data)

    def get_thread_id(self, user_id):
        # Retrieve thread_id for the given user_id
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'performance' in user_data:
            return user_data['performance'].get('thread_id')
        return None


    def add_assistant_id(self, user_id, assistant_id):
        # Add assistant_id to the user's performance data
        user_data = self.firestore_crud.read_document(user_id)
        if user_data:
            user_data['performance']['assistant_id'] = assistant_id
            self.firestore_crud.update_document(user_id, user_data)

    def get_assistant_id(self, user_id):
        # Retrieve assistant_id for the given user_id
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'performance' in user_data:
            return user_data['performance'].get('assistant_id')
        return None


    def add_or_update_score(self, user_id, score):
        # Add or update the score in the user's performance data
        user_data = self.firestore_crud.read_document(user_id)
        if user_data:
            user_data['performance']['score'] = score
            self.firestore_crud.update_document(user_id, user_data)

    def get_score(self, user_id):
        # Retrieve the score for the given user_id
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'performance' in user_data:
            return user_data['performance'].get('score')
        return None
    
    def add_or_update_chat(self, user_id, new_chat):
        # Retrieve existing user data from Firestore
        user_data = self.firestore_crud.read_document(user_id)

        if user_data:
            # Check if the 'chat' key exists and has an inner 'chat' list
            if 'chat' in user_data and 'chat' in user_data['chat']:
                # Append new chat messages to existing chat history
                user_data['chat']['chat'].extend(new_chat)
            else:
                # Create new chat structure if not present
                user_data['chat'] = {'chat': new_chat}

            # Update the document in Firestore with the updated chat data
            self.firestore_crud.update_document(user_id, user_data)
        else:
            # If no existing data, create new entry
            user_data = {'chat': {'chat': new_chat}}
            self.firestore_crud.create_document(user_id, user_data)


    
    def get_chat(self, user_id):
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'chat' in user_data:
            return user_data['chat'].get('chat')
        return None
    