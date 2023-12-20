import firebase_admin
from firebase_admin import credentials, firestore

class FirestoreCRUD:
    def __init__(self, credential_path, collection_name):
        cred = credentials.Certificate(credential_path)
        # firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.collection = self.db.collection(collection_name)

    def create_document(self, document_id, data):
        self.collection.document(document_id).set(data)

    def read_document(self, document_id):
        doc = self.collection.document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None

    def update_document(self, document_id, data):
        self.collection.document(document_id).update(data)

    def delete_document(self, document_id):
        self.collection.document(document_id).delete()
