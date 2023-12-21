from database.firestoreCRUD import FirestoreCRUD
from firebase_admin import credentials, firestore
import firebase_admin

class UserManager:
    def __init__(self):
        self.firestore_crud = FirestoreCRUD
        self.credential_path = '/home/codered/mystuff/progs/interview-mentor-firebase-adminsdk-sguq7-aee5c6cca8.json'
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
            user_data = {'details': details, 'performance': performance}
            self.firestore_crud.create_document(user_id, user_data)
