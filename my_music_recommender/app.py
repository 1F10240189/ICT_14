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
            error = f"Spotify検索でエラー: {e}"
            track_info = None

        if track_info:
            spotify_vec = spotify.features_to_vector(track_info.get("features", {}), dim=config.VECTOR_DIM)
            try:
                audio_source = track_info.get("preview_url")
                vggish_vec = extract_audio_vector(audio_source, dim=config.VGGISH_FEATURE_DIM)
            except Exception as e:
                print("audio vec error:", e)
                vggish_vec = np.zeros(config.VGGISH_FEATURE_DIM, dtype=np.float32)
            combined_vec = np.concatenate([spotify_vec, vggish_vec])
            try:
                results = vector_db.search(combined_vec, top_k=6)
            except Exception as e:
                error = f"ベクトル検索でエラー: {e}"
                results = []
            try:
                recommendation = generate_recommendation_text(track_info, results)
            except Exception as e:
                error = f"AI生成でエラー: {e}"
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