# app.py (最終修正版)
from flask import Flask, render_template, request, redirect, url_for, jsonify
from services.spotify_service import SpotifyService
from services.vggish_service import extract_audio_vector
from services.vector_db_service import VectorDB
from services.ai_agent_service import generate_recommendation_text
import config
import numpy as np

app = Flask(__name__)
spotify = SpotifyService()
vector_db = VectorDB(config.COMBINED_VECTORS_NPY, config.VECTORS_META)

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
            error = f"Spotifyから曲情報の取得に失敗しました: {e}"
            track_info = None

        if track_info:
            # 1. Spotify特徴量ベクトルを作成
            spotify_vec = spotify.features_to_vector(track_info.get("features", {}), dim=config.VECTOR_DIM)

            # 2. 音声ベクトル作成を試行
            vggish_vec = None
            try:
                audio_source = track_info.get("preview_url")
                if audio_source:
                    vggish_vec = extract_audio_vector(audio_source, dim=config.VGGISH_FEATURE_DIM)
                    print("成功: リアルタイム音声解析に成功しました。")
                else:
                    print("情報: この曲にはプレビュー音源がありません。")
            except Exception as e:
                # ★★★ 修正箇所 ★★★
                # ユーザーにエラーを通知し、何が起きているかを明確にする
                error = "警告: リアルタイムでの音声解析に失敗しました。PC環境によりダウンロードがブロックされた可能性があります。Spotify特徴量のみで推薦を続行します。"
                print(f"{error} (詳細: {e})")
            
            # 音声ベクトルが取得できなかった場合は、ゼロベクトルで埋める
            if vggish_vec is None:
                vggish_vec = np.zeros(config.VGGISH_FEATURE_DIM, dtype=np.float32)

            # 3. ベクトルを結合
            combined_vec = np.concatenate([spotify_vec, vggish_vec])

            # 4. データベース検索と推薦文生成
            try:
                results = vector_db.search(combined_vec, top_k=6)
                results = [r for r in results if r.get("id") != track_id][:5] # 自分自身を除外
                
                if not results:
                     # 既存のエラーメッセージと競合しないように、エラーがない場合のみ設定
                     if not error:
                        error = "類似曲が見つかりませんでした。"
                else:
                    recommendation = generate_recommendation_text(track_info, results)

            except Exception as e:
                error = f"推薦の生成中にエラーが発生しました: {e}"
                recommendation = None

    return render_template("index.html", recommendation=recommendation, error=error)

@app.route("/search_track", methods=["GET"])
def search_track():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"tracks": []})
    
    try:
        results = spotify.search_track_by_name(q, limit=5)
        return jsonify({"tracks": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
