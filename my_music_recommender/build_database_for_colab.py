# build_database_for_colab.py
# Google Colab上で実行するためのデータベース構築スクリプト

print("--- Google Colab用 データベース構築スクリプト ---")

# Colab環境のセットアップ
import os
import sys
import subprocess

print("ステップ1: 必要なライブラリをインストールします...")
# spotipy, librosa, numpy, tqdm をインストール
subprocess.check_call([sys.executable, "-m", "pip", "install", "spotipy", "librosa", "numpy", "tqdm", "soundfile"])
print(" -> ライブラリのインストールが完了しました。")

# --- モジュールのインポート ---
import numpy as np
import json
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import tempfile
import librosa
import soundfile as sf

# --- 設定 ---
# Spotify APIキー (ご自身のものに書き換えてください)
SPOTIPY_CLIENT_ID = "c0146d0eab3742b7bbf89c09c0e5588e"
SPOTIPY_CLIENT_SECRET = "58c0b49dafaf4757bc9145c6105afd00"

# ベクトルの次元数
VECTOR_DIM = 128
VGGISH_FEATURE_DIM = 128
COMBINED_VECTOR_DIM = VECTOR_DIM + VGGISH_FEATURE_DIM

# データベースのファイル名
VECTORS_NPY_FILENAME = "combined_track_vectors.npy"
META_JSON_FILENAME = "track_meta.json"

# SpotifyのプレイリストID (例: "37i9dQZEVXbKXQ4mDTEBXq" は "Top 50 - Global")
TARGET_PLAYLIST_ID = "37i9dQZEVXbKXQ4mDTEBXq"
TRACK_LIMIT = 100 # Colabなので100曲に挑戦

# --- VGGish音声分析関数 (vggish_service.pyから移植) ---
def _download_to_temp(url: str) -> str:
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp3")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return path

def extract_audio_vector(audio_source: str, sr: int = 22050, n_mels: int = 96, dim=VGGISH_FEATURE_DIM) -> np.ndarray:
    temp_file = None
    try:
        if audio_source.startswith("http"):
            temp_file = _download_to_temp(audio_source)
            path = temp_file
        else:
            path = audio_source
        y, _ = librosa.load(path, sr=sr, mono=True, duration=30.0)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        v = np.mean(S_db, axis=1)
        if v.shape[0] >= dim:
            out = v[:dim]
        else:
            pad = np.zeros(dim - v.shape[0], dtype=float)
            out = np.concatenate([v, pad])
        out = (out - np.mean(out)) / (np.std(out) + 1e-9)
        out = out / (np.linalg.norm(out) + 1e-9)
        return out.astype(np.float32)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

# --- Spotify API連携関数 (spotify_service.pyから移植) ---
auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_track_info(track_id: str) -> dict:
    t = sp.track(track_id)
    af = sp.audio_features([track_id])[0] or {}
    return {"id": track_id, "name": t["name"], "artist": t["artists"][0]["name"], "preview_url": t.get("preview_url"), "features": af}

def features_to_vector(features: dict, dim=VECTOR_DIM) -> np.ndarray:
    keys = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    vals = []
    for k in keys:
        v = features.get(k, 0.0) if features else 0.0
        if k == "tempo" and v is not None: v = v / 200.0
        vals.append(v or 0.0)
    arr = np.array(vals, dtype=float)
    if arr.size >= dim: return arr[:dim]
    else: return np.concatenate([arr, np.zeros(dim - arr.size)])

# --- メイン処理 ---
def build_database():
    print(f"\nステップ2: プレイリスト '{TARGET_PLAYLIST_ID}' から最大 {TRACK_LIMIT} 曲の情報を取得します...")
    try:
        playlist = sp.playlist_items(TARGET_PLAYLIST_ID, limit=TRACK_LIMIT)
        track_items = playlist["items"]
        print(f" -> {len(track_items)} 曲の情報を取得しました。")
    except Exception as e:
        print(f"!!! エラー: Spotify APIからのプレイリスト取得に失敗しました: {e}")
        return

    all_vectors, all_meta = [], []
    print(f"\nステップ3: {len(track_items)} 曲のベクトル（Spotify特徴量 + VGGish音声分析）を生成します...")
    for item in tqdm(track_items, desc="楽曲処理中"):
        track = item.get("track")
        if not (track and track.get("id") and track.get("preview_url")):
            continue
        try:
            info = get_track_info(track["id"])
            spotify_vec = features_to_vector(info["features"])
            vggish_vec = extract_audio_vector(info["preview_url"])
            combined_vec = np.concatenate([spotify_vec, vggish_vec])
            all_vectors.append(combined_vec)
            all_meta.append({"id": info["id"], "name": info["name"], "artist": info["artist"]})
        except Exception as e:
            tqdm.write(f"エラー: \"{track.get('name', '不明')}\" の処理で失敗。スキップします。(詳細: {e})")

    if not all_vectors:
        print("\n!!! 警告: 有効なベクトルが生成されませんでした。")
        return

    vectors_np = np.array(all_vectors, dtype=np.float32)
    print(f"\nステップ4: データベースファイルを保存します...")
    np.save(VECTORS_NPY_FILENAME, vectors_np)
    print(f" -> ベクトルファイル (形状: {vectors_np.shape}) を '{VECTORS_NPY_FILENAME}' に保存しました。")
    with open(META_JSON_FILENAME, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f" -> メタ情報 ({len(all_meta)}件) を '{META_JSON_FILENAME}' に保存しました。")
    print("\n★★★ データベースの構築が完了しました！ ★★★")
    print("画面左のファイル一覧から、上記2つのファイルをダウンロードして、ローカルPCの 'data' フォルダに配置してください。")

# --- 実行 ---
build_database()