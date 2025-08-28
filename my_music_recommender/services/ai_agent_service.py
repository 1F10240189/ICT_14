# my_music_recommender/services/ai_agent_service.py (最終版)
import os
import openai
import config
import markdown

openai.api_key = config.OPENAI_API_KEY

def _format_features(features):
    if not features: return "特徴量データなし"
    items = [f"{k}: {features.get(k):.2f}" for k in ["danceability", "energy", "valence"] if features.get(k)]
    if features.get("tempo"): items.append(f"テンポ: {features.get('tempo'):.0f} BPM")
    return ", ".join(items) if items else "詳細な特徴量なし"

def _make_prompt(track_info, similar_tracks):
    lines = [
        f"■元の曲: \"{track_info.get('name')}\" - {track_info.get('artist')}",
        f"  (特徴: {_format_features(track_info.get('features'))})",
        "", "■似ている曲の候補リスト:"
    ]
    for i, s in enumerate(similar_tracks, 1):
        lines.append(f"{i}. \"{s.get('name')}\" - {s.get('artist')} (類似スコア: {s.get('score'):.3f})")
    
    lines.extend([
        "", "---", "■指示:", "あなたはプロの音楽評論家です。",
        "上記の情報を基に、以下の形式で推薦文を日本語で作成してください。",
        "1. **総評**: なぜこれらの曲がおすすめなのか、「元の曲」の音楽的特徴と関連付けながら、100〜150文字で魅力的に説明してください。",
        "2. **各曲へのコメント**: 候補の各曲について、どのような点が似ているか、あるいは違いを楽しめるかを、それぞれ1行で簡潔にコメントしてください。"
    ])
    return "\\n".join(lines)

def generate_recommendation_text(track_info, similar_tracks):
    if not config.OPENAI_API_KEY:
        return f"おすすめ候補（簡易）: " + ", ".join([f"{t['name']}（{t['artist']}）" for t in similar_tracks])

    prompt = _make_prompt(track_info, similar_tracks)
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":"あなたは音楽の専門家として、ユーザーにパーソナライズされた楽曲推薦を行います。"},
            {"role":"user", "content": prompt}
        ],
        max_tokens=500, temperature=0.7,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    return markdown.markdown(text)