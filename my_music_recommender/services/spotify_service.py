import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import config
import requests
import os
import json
from typing import Optional

class SpotifyService:
    def __init__(self):
        client_id = config.SPOTIPY_CLIENT_ID
        client_secret = config.SPOTIPY_CLIENT_SECRET
        if not client_id or not client_secret:
            raise RuntimeError("SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET が設定されていません。")
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def search_track_by_name(self, q: str) -> dict:
        """
        曲名や "artist - track" を投げると、見つかった最初のトラック情報を返す。
        戻り値の例:
        {
          "id": "...",
          "name": "...",
          "artist": "...",
          "preview_url": "...",  # 30秒プレビューがあれば
          "features": {...}      # audio_features からの特徴量
        }
        """
        res = self.sp.search(q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            raise ValueError("曲が見つかりませんでした。")
        t = items[0]
        track_id = t["id"]
        preview_url = t.get("preview_url")  # 30sのプレビューURL。無い場合あり
        name = t["name"]
        artist = t["artists"][0]["name"]
        # audio features
        af = self.sp.audio_features([track_id])[0] or {}
        return {
            "id": track_id,
            "name": name,
            "artist": artist,
            "preview_url": preview_url,
            "features": af
        }

    def download_preview(self, preview_url: str, dst_path: str) -> str:
        """
        preview_url（Spotifyの30秒mp3等）をダウンロードしてローカルに保存
        """
        if not preview_url:
            raise ValueError("preview_urlがありません。")
        r = requests.get(preview_url, stream=True, timeout=10)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return dst_path

    def features_to_vector(self, features: dict, dim=None):
        """
        Spotify が返す audio_features をベクトルにマッピングする簡易関数。
        VGGishベクトルが取得できないときのフォールバックとして使用。
        """
        import numpy as np
        if dim is None:
            import config
            dim = config.VECTOR_DIM
        # 代表的な数値指標を抽出
        keys = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
        vals = []
        for k in keys:
            v = features.get(k, 0.0)
            # tempoはスケールが大きいので正規化
            if k == "tempo":
                v = v / 200.0
            vals.append(v)
        # パディングまたはランダムで埋めて固定長にする
        arr = np.array(vals, dtype=float)
        if arr.size >= dim:
            return arr[:dim]
        pad = np.random.RandomState(0).rand(dim - arr.size) * 0.01
        return np.concatenate([arr, pad])
