from database.firestoreCRUD import FirestoreCRUD
from firebase_admin import credentials, firestore
import firebase_admin
from security.password_manager import Password_manager

class UserManager:
    def __init__(self):
        self.firestore_crud = FirestoreCRUD
        self.credential_path = 'database/interview-mentor-firebase-adminsdk-sguq7-aee5c6cca8.json'
        self.collection_name = 'users'
        self.password_manager = Password_manager()
        if not firebase_admin._apps:
            cred = credentials.Certificate(self.credential_path)
            firebase_admin.initialize_app(cred)

        self.firestore_crud = FirestoreCRUD(self.credential_path, self.collection_name)
    def initialize_user(self, user_id, name, email, password="_"):
        user_data = self.firestore_crud.read_document(user_id)
        if not user_data:
            # hash the password
            password = self.password_manager.hash_password(password)
            # User does not exist, so create new user
            details = {'name': name, 'email': email, 'password': password}
            performance = {'score': []}  
            chat = {}
            preference={}
            
            user_data = {'details': details, 'performance': performance, 'chat': chat, 'preference': preference}
            self.firestore_crud.create_document(user_id, user_data)

    def is_user(self, user_id):
        user_data = self.firestore_crud.read_document(user_id)
        if not user_data:
            return False
        else:
            return True
        
    def check_password(self, user_id, password):
        user_data = self.firestore_crud.read_document(user_id)
        stored_hash = user_data['details'].get('password')

        return self.password_manager.verify_password(stored_hash, password)

    


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
        scores = user_data['performance']['score']
        scores.append(score)
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
            # Replace the 'chat' key with the new chat data, regardless of previous content
            user_data['chat'] = {'chat': new_chat}
            # Update the document in Firestore with the new chat data
            self.firestore_crud.update_document(user_id, user_data)
        else:
            # If no existing data, create new entry with the new chat
            user_data = {'chat': {'chat': new_chat}}
            self.firestore_crud.create_document(user_id, user_data)

    def add_or_update_interviewer(self, user_id, interviewer):
        user_data = self.firestore_crud.read_document(user_id)
        if user_data:
            user_data['preference']['interviewer'] = interviewer
            self.firestore_crud.update_document(user_id, user_data)
    def add_or_update_language(self, user_id, language):
        user_data = self.firestore_crud.read_document(user_id)
        if user_data:
            user_data['preference']['language'] = language
            self.firestore_crud.update_document(user_id, user_data)
    def get_interviewer(self, user_id):
        # Retrieve the score for the given user_id
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'preference' in user_data:
            return user_data['preference'].get('interviewer')
        return None
    
    def get_name(self, user_id):
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'details' in user_data:
            return user_data['details'].get('name')
        return None

    def get_language(self, user_id):
        # Retrieve the score for the given user_id
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'preference' in user_data:
            return user_data['preference'].get('language')
        return None
    
    def add_or_update_user_setting(self, user_id, interviewer, language):
        self.add_or_update_interviewer(user_id, interviewer=interviewer)
        self.add_or_update_language(user_id, language=language)

    def  get_user_setting(self, user_id):
        preference = {"interviewer": "", "language": ""}
        preference['interviewer'] = self.get_interviewer(user_id)
        preference['language'] = self.get_language(user_id)
        return preference

    
    def get_chat(self, user_id):
        user_data = self.firestore_crud.read_document(user_id)
        if user_data and 'chat' in user_data:
            return user_data['chat'].get('chat')
        return None
    