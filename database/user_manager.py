class UserManager:
    def __init__(self, firestore_crud):
        self.firestore_crud = firestore_crud

    def initialize_user(self, user_id, name, email):
        user_data = self.firestore_crud.read_document(user_id)
        if not user_data:
            # User does not exist, so create new user
            details = {'name': name, 'email': email}
            performance = {}  # Initialize performance data, if any
            user_data = {'details': details, 'performance': performance}
            self.firestore_crud.create_document(user_id, user_data)
