"""
音声特徴量抽出モジュール。

※ 本来は Google の VGGish（学習済みモデル）を使うのが望ましいですが、
  環境によって導入が面倒なため、ここでは librosa を使った簡易実装を載せます。
  精度を上げたい場合は、VGGish の出力 (128-d embedding) に差し替えてください。
"""

import numpy as np
import librosa
import requests
import os
import tempfile
from typing import Optional
import soundfile as sf

def _download_to_temp(url: str) -> str:
    """HTTP URL を一時ファイルにダウンロードしてパスを返す"""
    if url.startswith("http"):
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return path
    return url

def extract_audio_vector(audio_source: Optional[str], sr: int = 22050, n_mels: int = 96, dim=None) -> np.ndarray:
    """
    audio_source: ローカルファイルパス or HTTP URL (mp3/ogg/wav)
    戻り値: dim 次元の numpy ベクトル
    """
    if dim is None:
        import config
        dim = config.VGGISH_FEATURE_DIM
    # 1) 入力ファイルをローカルに用意
    temp_file = None
    try:
        if audio_source is None:
            raise ValueError("audio_sourceが指定されていません。")
        if audio_source.startswith("http"):
            temp_file = _download_to_temp(audio_source)
            path = temp_file
        else:
            path = audio_source
        # 2) librosaで読み込み
        y, _ = librosa.load(path, sr=sr, mono=True, duration=30.0)  # previewは30s
        # 3) メルスペクトログラム
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        # 4) time方向で平均を取り、n_mels次元のベクトルにする
        v = np.mean(S_db, axis=1)
        # 5) 正規化と次元圧縮（簡易：FFTで高次を取る or PCA）
        # ここでは簡易に線形変換で dim に合わせる
        if v.shape[0] >= dim:
            out = v[:dim]
        else:
            pad = np.zeros(dim - v.shape[0], dtype=float)
            out = np.concatenate([v, pad])
        # 標準化
        out = (out - np.mean(out)) / (np.std(out) + 1e-9)
        # L2正規化
        out = out / (np.linalg.norm(out) + 1e-9)
        return out.astype(np.float32)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
