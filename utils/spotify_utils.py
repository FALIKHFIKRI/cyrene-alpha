"""
Spotify utilities (playlist functions removed)

This module used to provide helpers for creating Spotify playlists and searching tracks.
Those functions were removed per the user's request. Importing this module is still safe,
but attempting to use the removed helpers will raise an informative error.
"""

def not_implemented(*args, **kwargs):
    raise NotImplementedError("Playlist-related Spotify utilities were removed.")

is_spotipy_available = not_implemented
create_spotify_client = not_implemented
search_track_uri = not_implemented
create_playlist = not_implemented
