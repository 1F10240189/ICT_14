from flask import Flask, render_template, request
from services.spotify_service import get_track_info
from services.vggish_service import extract_audio_vector
from services.vector_db_service import search_vector_db
from services.ai_agent_service import generate_recommendation_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        track_name = request.form["track_name"]
        # Spotify情報取得
        track_info = get_track_info(track_name)
        # 音声特徴量抽出
        audio_vec = extract_audio_vector(track_info.audio_file)
        # 類似曲検索
        similar_tracks = search_vector_db(audio_vec)
        # AI推薦文作成
        recommendation = generate_recommendation_text(similar_tracks)
        return render_template("index.html", recommendation=recommendation)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

