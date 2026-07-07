from __future__ import annotations

import json
from pathlib import Path

import ee
from config import settings
from google.oauth2.credentials import Credentials


def get_ee_credentials() -> Credentials:
    """
    Build Earth Engine OAuth credentials from a refresh-token file.

    If settings.ee_credentials_path (env var EE_CREDENTIALS_PATH) is set,
    reads the token from that custom path. Otherwise falls back to
    ee.data.get_persistent_credentials(), which reads the default
    ~/.config/earthengine/credentials.

    Earth Engine's own get_credentials_path() is hardcoded to
    ~/.config/earthengine/credentials with no env-var override (verified
    against the earthengine-api source — this has been a standing feature
    request since 2018 that was never added). So a custom path has to be
    handled here: read the refresh token ourselves and reassemble the
    Credentials object with ee's bundled OAuth client_id/client_secret/
    token_uri — the same fields get_persistent_credentials() merges in
    internally. Without this merge, google-auth's Credentials.refresh()
    fails with "The credentials do not contain the necessary fields...",
    since the token file on its own only ever contains a refresh_token.
    """
    if settings.ee_credentials_path is None:
        return ee.data.get_persistent_credentials()

    path = Path(settings.ee_credentials_path)
    with open(path) as f:
        info = json.load(f)

    if "refresh_token" not in info:
        raise ValueError(
            f"{path} has no 'refresh_token' key — expected the file "
            f"written by `ee.Authenticate()` (or `earthengine authenticate`)."
        )

    return Credentials(
        None,
        refresh_token=info["refresh_token"],
        token_uri=ee.oauth.TOKEN_URI,
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=ee.oauth.SCOPES,
    )
