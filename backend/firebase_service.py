import os
import firebase_admin
from firebase_admin import credentials, messaging

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"

if not firebase_admin._apps:
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise FileNotFoundError(f"Firebase key file not found: {SERVICE_ACCOUNT_PATH}")

    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)


def send_notification(token: str, title: str, body: str, data: dict = None):
    """
    Send push notification to one device token
    """
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        data=data or {},
        token=token
    )

    response = messaging.send(message)
    print(f"✅ Notification sent: {response}")
    return response