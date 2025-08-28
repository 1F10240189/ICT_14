# app.py (ハイブリッド検索対応・最終完成版)
from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.spotify_service import SpotifyService
from services.vector_db_service import VectorDB
from services.ai_agent_service import generate_recommendation_text
import config
import numpy as np

app = Flask(__name__)
spotify = SpotifyService()
vector_db = VectorDB(config.COMBINED_VECTORS_NPY, config.VECTORS_META)
DB_META_DICT = {item['id']: idx for idx, item in enumerate(vector_db.meta or [])}

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = None
    error = None
    if request.method == "POST":
        track_id = request.form.get("track_id", "").strip()
        if not track_id:
            return redirect(url_for("index"))

        try:
            track_info = spotify.get_track_info(track_id)
        except Exception as e:
            return render_template("index.html", error=f"Spotifyから曲情報の取得に失敗しました: {e}")

        if not track_info:
            return render_template("index.html", error="曲情報が見つかりませんでした。")

        results = []
        
        # --- ▼▼▼ ハイブリッド検索ロジック ▼▼▼ ---
        # 1. ローカルDBに曲が存在する場合 -> VGGishを含む高精度な「完全一致検索」
        if track_id in DB_META_DICT:
            mode_message = "【高精度モード】VGGishの音声分析情報を含むベクトルで推薦します。"
            print(f"情報: ローカルDB内の曲 '{track_info['name']}' が選択されました。{mode_message}")
            idx = DB_META_DICT[track_id]
            query_vec = vector_db.vectors[idx]
            results = vector_db.search_combined(query_vec, top_k=6)

        # 2. ローカルDBにない曲の場合 -> VGGishを使わない「Spotify特徴量検索」
        else:
            if track_info.get("features"):
                mode_message = "【通常モード】Spotifyの楽曲特徴のみで推薦します。（お客様の環境ではリアルタイム音声分析がブロックされるため）"
                print(f"情報: 新しい曲 '{track_info['name']}' が選択されました。{mode_message}")
                query_vec = spotify.features_to_vector(track_info["features"], dim=config.VECTOR_DIM)
                results = vector_db.search_spotify_only(query_vec, top_k=6)
            else:
                error = "この曲は楽曲特徴量が取得できないため、推薦を生成できませんでした。"
        # --- ▲▲▲ ハイブリッド検索ロジックここまで ▲▲▲ ---

        if not error and not results:
            error = "類似曲が見つかりませんでした。"
        elif results:
            try:
                # 検索結果から自分自身を除外
                final_results = [r for r in results if r.get("id") != track_id][:5]
                if final_results:
                    recommendation = generate_recommendation_text(track_info, final_results)
                else:
                    error = "類似曲が見つかりませんでした。（検索結果が選択した曲自身のみでした）"
            except Exception as e:
                error = f"AIによる推薦文の生成中にエラーが発生しました: {e}"

    return render_template("index.html", recommendation=recommendation, error=error)

@app.route("/search_track", methods=["GET"])
def search_track():
    q = request.args.get("q", "").strip()
    if not q: return jsonify({"tracks": []})
    try:
        return jsonify({"tracks": spotify.search_track_by_name(q, limit=5)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)