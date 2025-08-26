import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import config
import requests
import os
import json
from typing import Optional, Dict, List
import numpy as np
import urllib.parse

class SpotifyService:
    def __init__(self):
        client_id = config.SPOTIPY_CLIENT_ID
        client_secret = config.SPOTIPY_CLIENT_SECRET
        if not client_id or not client_secret:
            raise RuntimeError("SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET が設定されていません。")
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def search_track_by_name(self, q: str, limit: int = 5) -> List[Dict]:
        """
        曲名や "artist - track" を投げると、複数のトラック候補を返す。
        """
        if not isinstance(q, str):
            q = str(q)
        
        try:
            res = self.sp.search(q, type="track", limit=limit)
            items = res.get("tracks", {}).get("items", [])
            if not items:
                return []
            
            results = []
            for t in items:
                results.append({
                    "id": t["id"],
                    "name": t["name"],
                    "artist": t["artists"][0]["name"],
                    "album_art": t["album"]["images"][0]["url"] if t["album"]["images"] else None
                })
            return results
        except spotipy.exceptions.SpotifyException as e:
            raise RuntimeError(f"Spotify APIエラー: {e}") from e

    def get_track_info(self, track_id: str) -> Dict:
        """
        トラックIDを使って、1曲の詳細情報を取得する。
        """
        try:
            t = self.sp.track(track_id)
            preview_url = t.get("preview_url")
            
            # ★★★ 修正箇所：audio_featuresの取得に失敗した場合のエラー処理を追加 ★★★
            try:
                af = self.sp.audio_features([track_id])[0] or {}
            except spotipy.exceptions.SpotifyException as e:
                print(f"警告: audio_featuresの取得に失敗しました。{e}")
                af = {} # 失敗した場合は空の辞書を返す
            
            return {
                "id": track_id,
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "preview_url": preview_url,
                "features": af
            }
        except spotipy.exceptions.SpotifyException as e:
            raise RuntimeError(f"Spotify APIエラー: {e}") from e

    def download_preview(self, preview_url: str, dst_path: str) -> str:
        """
        preview_url（Spotifyの30秒mp3等）をダウンロードしてローカルに保存
        """
        if not preview_url:
            raise ValueError("preview_urlがありません。")
        try:
            r = requests.get(preview_url, stream=True, timeout=10)
            r.raise_for_status()
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return dst_path
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ダウンロードエラー: {e}") from e

    def features_to_vector(self, features: Dict, dim: Optional[int] = None) -> np.ndarray:
        """
        Spotify が返す audio_features をベクトルにマッピングする簡易関数。
        VGGishベクトルが取得できないときのフォールバックとして使用。
        """
        if dim is None:
            dim = config.VECTOR_DIM
        
        # 代表的な数値指標を抽出
        keys = ["danceability", "energy", "speechiness", "acousticness", 
                "instrumentalness", "liveness", "valence", "tempo"]
        vals = []
        for k in keys:
            v = features.get(k, 0.0)
            if k == "tempo":
                v = v / 200.0
            vals.append(v)
            
        arr = np.array(vals, dtype=float)
        
        if arr.size >= dim:
            return arr[:dim]
        else:
            pad = np.random.RandomState(0).rand(dim - arr.size) * 0.01
            return np.concatenate([arr, pad])