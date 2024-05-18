from google.cloud import storage
import os

class Handler:
    def __init__(self):
        self.BUCKET_NAME = 'asia.artifacts.interview-mentor-408213.appspot.com'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'interview-mentor-408213-f5ba84c00ba7.json'

    def upload_to_gcs(self, file, filename):
        """uploads file to Google Cloud Storage."""
        try:
            client = storage.Client()
            bucket = client.get_bucket(self.BUCKET_NAME)
            blob = bucket.blob(filename)
            blob.upload_from_file(file, content_type=file.content_type)
            log = f"File uploaded to GCS: {filename}"
            print(log)
            return f"gs://{self.BUCKET_NAME}/{filename}", log
        except Exception as e:
            log = f"Failed to upload file to GCS: {str(e)}"
            print(log)
            return None, log

    def delete_from_gcs(self, filename):
        """deletes file from Google Cloud Storage."""
        client = storage.Client()
        bucket = client.bucket(self.BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.delete()