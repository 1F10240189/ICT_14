# vector_db_service.py (ハイブリッド検索対応版)
import os
import json
import numpy as np
import config

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors

class VectorDB:
    def __init__(self, vectors_path, meta_path):
        self.vectors_path = vectors_path
        self.meta_path = meta_path
        self.vectors = None
        self.meta = None
        
        # 2種類の検索インデックスを準備
        self.combined_index = None # (Spotify + VGGish) のインデックス
        self.spotify_only_index = None # (Spotifyのみ) のインデックス

        self._load()

    def _load(self):
        if not (os.path.exists(self.vectors_path) and os.path.exists(self.meta_path)):
            print("ベクトルファイルが見つからないため、処理を中断します。")
            print("まず install_prebuilt_db.py を実行してください。")
            return

        self.vectors = np.load(self.vectors_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        self._build_all_indexes()

    def _build_all_indexes(self):
        # 1. 完全ベクトル用のインデックスを構築
        dim_combined = config.COMBINED_VECTOR_DIM
        if self.vectors.shape[1] != dim_combined:
            raise ValueError(f"読み込まれたベクトルの次元数({self.vectors.shape[1]})が設定({dim_combined})と異なります。")
        
        if _HAS_FAISS:
            self.combined_index = faiss.IndexFlatL2(dim_combined)
            self.combined_index.add(self.vectors.astype(np.float32))
        else:
            self.combined_index = NearestNeighbors(n_neighbors=10, metric="euclidean")
            self.combined_index.fit(self.vectors)

        # 2. Spotify特徴量のみのベクトル用のインデックスを構築
        dim_spotify = config.VECTOR_DIM
        spotify_vectors = self.vectors[:, :dim_spotify].copy()
        if _HAS_FAISS:
            self.spotify_only_index = faiss.IndexFlatL2(dim_spotify)
            self.spotify_only_index.add(spotify_vectors.astype(np.float32))
        else:
            self.spotify_only_index = NearestNeighbors(n_neighbors=10, metric="euclidean")
            self.spotify_only_index.fit(spotify_vectors)
        print(" -> 2種類の検索インデックス（完全版とSpotify限定版）の構築が完了しました。")

    def _search_with_index(self, index, query_vector, top_k):
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if _HAS_FAISS:
            D, I = index.search(q, top_k)
            scores, idxs = D[0].tolist(), I[0].tolist()
        else:
            dists, idxs = index.kneighbors(q, n_neighbors=top_k)
            scores, idxs = dists[0].tolist(), idxs[0].tolist()
        
        results = []
        for i, s in zip(idxs, scores):
            if 0 <= i < len(self.meta):
                m = self.meta[i].copy()
                m["score"] = float(s)
                results.append(m)
        return results

    def search_combined(self, query_vector, top_k=5):
        """VGGish情報を含む完全なベクトルで検索する"""
        if not self.combined_index: return []
        return self._search_with_index(self.combined_index, query_vector, top_k)

    def search_spotify_only(self, query_vector, top_k=5):
        """Spotify特徴量のみのベクトルで検索する"""
        if not self.spotify_only_index: return []
        return self._search_with_index(self.spotify_only_index, query_vector, top_k)