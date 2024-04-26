import firebase_admin
from firebase_admin import credentials, storage, firestore


class Database:
    def __init__(self):
        # Replace the projectID eg.
        self.bucket_name = 'testone-b55de.appspot.com'
        # self.bucket_name = '<projectID>.appspot.com'

        # You need to download the serviceaccount.json
        self.fb_cred = 'service_account.json'
        cred = credentials.Certificate(self.fb_cred)
        firebase_admin.initialize_app(cred,
                                      {'storageBucket': self.bucket_name})
        self.db = firestore.client()  # this connects to our Firestore database

    def set_events(self, collection: str, device: str, data: dict):
        collection = self.db.collection(collection)  # opens collection
        doc = collection.document(device)  # specifies the  document
        doc.set(data, merge=True)

    def exists_on_cloud(self, filename):
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        if blob.exists():
            return blob.public_url
        else:
            return False

    def upload_file(self, firebase_path, local_path):
        bucket = storage.bucket()
        blob = bucket.blob(firebase_path)
        blob.upload_from_filename(local_path)
        print('This file is uploaded to cloud.')
        blob.make_public()
        url = blob.public_url
        return url
