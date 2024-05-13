import requests
import base64
from dotenv import load_dotenv
import os


class Spotify:
    def __init__(self):
        # Spotify API endpoints
        self.token_endpoint = "https://accounts.spotify.com/api/token"
        self.now_playing_endpoint = "https://api.spotify.com/v1/me/player/currently-playing"


    def get_access_token(self, client_id, client_secret, refresh_token):
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode("utf-8")
        headers = {
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        body = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        response = requests.post(self.token_endpoint, headers=headers, data=body)
        return response.json().get('access_token')



    def get_currently_playing(self, access_token):
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.get(self.now_playing_endpoint, headers=headers)
        if response.status_code == 200:
            track = response.json()
            if track['currently_playing_type'] == 'ad': #ad running
                return "not playing"
            if track and 'item' in track and track['item']:
                return {
                    'song_name': track['item']['name'],
                    'artist': ', '.join(artist['name'] for artist in track['item']['artists']),
                    'album': track['item']['album']['name']
                }
        elif response.status_code == 204:
            return "not playing"
        else:
            return f"Error: {response.status_code}"


    def get_image_url(self, access_token):
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.get(self.now_playing_endpoint, headers=headers)
        if response.status_code == 200:
            track = response.json()

            if track and 'item' in track and track['item']:
                images = track['item']['album']['images']
                for item in images:
                    if item['height'] == 300:
                        return(item['url'])

        elif response.status_code == 204: #not playing
            return None 
        else:
            return None


    def get_song(self, client_id, client_secret, refresh_token):
        access_token = self.get_access_token(client_id, client_secret, refresh_token)
        currently_playing = self.get_currently_playing(access_token)
        image_url = self.get_image_url(access_token)
        return currently_playing, image_url
