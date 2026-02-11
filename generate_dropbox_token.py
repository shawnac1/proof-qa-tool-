#!/usr/bin/env python3
"""
Dropbox Refresh Token Generator for Proof by Aerial Canvas

Run this script to generate a new refresh token with all required permissions.
"""

import webbrowser
import urllib.parse

# Your Dropbox app credentials
APP_KEY = "ta9b2km6af2j2h4"
APP_SECRET = "wknntqselzkuqof"

# Required scopes for the Photo Sort feature
SCOPES = [
    "files.metadata.read",
    "files.metadata.write",
    "files.content.read",
    "files.content.write",
    "sharing.read",
    "sharing.write"
]

def main():
    print("\n" + "="*60)
    print("Dropbox Refresh Token Generator")
    print("="*60)

    # Build the authorization URL
    auth_url = (
        f"https://www.dropbox.com/oauth2/authorize"
        f"?client_id={APP_KEY}"
        f"&response_type=code"
        f"&token_access_type=offline"
        f"&scope={' '.join(SCOPES)}"
    )

    print("\nStep 1: Opening your browser to authorize the app...")
    print(f"\nIf the browser doesn't open, go to this URL:\n{auth_url}\n")

    webbrowser.open(auth_url)

    print("\nStep 2: After authorizing, Dropbox will show you an ACCESS CODE.")
    print("Copy that code and paste it below:\n")

    auth_code = input("Enter the access code: ").strip()

    if not auth_code:
        print("No code entered. Exiting.")
        return

    print("\nStep 3: Exchanging code for refresh token...")

    import requests

    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "code": auth_code,
            "grant_type": "authorization_code",
            "client_id": APP_KEY,
            "client_secret": APP_SECRET
        }
    )

    if response.status_code == 200:
        data = response.json()
        refresh_token = data.get("refresh_token")
        access_token = data.get("access_token")

        print("\n" + "="*60)
        print("SUCCESS! Here's your new refresh token:")
        print("="*60)
        print(f"\n{refresh_token}\n")
        print("="*60)
        print("\nCopy this token and give it to Claude to update qa_tool.py")
        print("="*60 + "\n")

    else:
        print(f"\nError: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
