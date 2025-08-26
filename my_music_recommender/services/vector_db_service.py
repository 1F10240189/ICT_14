"""
ベクトルDB を扱うモジュール。
- faiss があればそれを使って高速検索
- 無ければ scikit-learn の NearestNeighbors にフォールバック

想定: データフォルダに track_vectors.npy (N x D) と track_meta.json (N 個のメタ情報) を置く。
"""

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
        self.index = None
        self.vectors = None
        self.meta = None
        self.dim = config.COMBINED_VECTOR_DIM  # 結合ベクトルの次元数に変更
        self._load()

    def _load(self):
        if os.path.exists(self.vectors_path) and os.path.exists(self.meta_path):
            self.vectors = np.load(self.vectors_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self._build_index()
        else:
            # 初期化用のダミーデータ（デモ目的）
            print("ベクトルファイルが見つかりません。デモ用の小さなDBを作成します。")
            # create a tiny demo DB
            self.meta = [
                {"id":"demo_1","name":"曲A","artist":"artistA"},
                {"id":"demo_2","name":"曲B","artist":"artistB"},
                {"id":"demo_3","name":"曲C","artist":"artistC"},
            ]
            self.vectors = np.random.RandomState(0).rand(len(self.meta), self.dim).astype(np.float32)
            os.makedirs(os.path.dirname(self.vectors_path), exist_ok=True)
            np.save(self.vectors_path, self.vectors)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False, indent=2)
            self._build_index()

    def _build_index(self):
        if _HAS_FAISS:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(self.vectors)
        else:
            self.index = NearestNeighbors(n_neighbors=10, algorithm="auto", metric="euclidean")
            self.index.fit(self.vectors)

    def search(self, query_vector, top_k=5):
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if _HAS_FAISS:
            D, I = self.index.search(q, top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            dists, idxs = self.index.kneighbors(q, n_neighbors=top_k, return_distance=True)
            idxs = idxs[0].tolist()
            scores = dists[0].tolist()
        results = []
        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(self.meta):
                continue
            m = self.meta[i].copy()
            m["score"] = float(s)
            results.append(m)
        return results

    def add(self, vector, meta):
        """
        デモ用：新曲をDBに追加して保存する（簡易実装）
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(1,-1)
        self.vectors = np.vstack([self.vectors, vec])
        self.meta.append(meta)
        np.save(self.vectors_path, self.vectors)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        self._build_index()
