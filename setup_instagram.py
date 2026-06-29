import os
from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import ChallengeRequired, TwoFactorRequired

load_dotenv()

USERNAME = os.getenv("INSTAGRAM_USERNAME", "").strip()
PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "").strip()
SESSION_FILE = "instagram_session.json"

cl = Client()

try:
    cl.login(USERNAME, PASSWORD)
    cl.dump_settings(SESSION_FILE)
    print(f"Login successful! Session saved to {SESSION_FILE}")

except ChallengeRequired:
    print("Instagram sent a verification code to your phone or email.")
    code = input("Enter the code here: ").strip()
    cl.challenge_resolve(cl.last_challenge)
    cl.dump_settings(SESSION_FILE)
    print(f"Verified! Session saved to {SESSION_FILE}")

except TwoFactorRequired:
    print("2FA is enabled on your account.")
    code = input("Enter your 2FA code: ").strip()
    cl.login(USERNAME, PASSWORD, verification_code=code)
    cl.dump_settings(SESSION_FILE)
    print(f"Login successful! Session saved to {SESSION_FILE}")

except Exception as e:
    print(f"Login failed: {e}")
