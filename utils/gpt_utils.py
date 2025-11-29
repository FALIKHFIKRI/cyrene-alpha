"""
GPT utilities (playlists removed)

This module previously contained playlist-related GPT helpers. Playlist functionality
has been removed per the user's request, so this module intentionally has no active
functions. Importing it is still safe but will not provide playlist behavior.
"""

def not_implemented(*args, **kwargs):
    """Raise an informative error if playlist-related GPT helpers are called."""
    raise NotImplementedError("Playlist-related GPT utilities were removed.")

# Kept symbol for compatibility if imported elsewhere; calling will raise.
is_openai_available = not_implemented
generate_playlist_with_gpt = not_implemented
