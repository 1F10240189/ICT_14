import os

# Spotify (Client Credentials flow)
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIPY_CLIENT_ID = "あなたのSpotify Client ID"
SPOTIPY_CLIENT_SECRET = "あなたのSpotify Client Secret"
# OpenAI (or other LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Vector DB / features
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "128"))

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRACKS_JSON = os.path.join(DATA_DIR, "sample_tracks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "track_vectors.npy")
VECTORS_META = os.path.join(DATA_DIR, "track_meta.json")

# VGGish音声ベクトルの保存先
VGGISH_VECTORS_NPY = os.path.join(DATA_DIR, "vggish_vectors.npy")
VGGISH_FEATURE_DIM = int(os.getenv("VGGISH_FEATURE_DIM", "128"))  # VGGishベクトルの次元数

# Spotify特徴量とVGGishベクトルを結合したベクトルの保存先
COMBINED_VECTORS_NPY = os.path.join(DATA_DIR, "combined_track_vectors.npy")
COMBINED_VECTOR_DIM = VECTOR_DIM + VGGISH_FEATURE_DIM  # 結合後のベクトル次元数
