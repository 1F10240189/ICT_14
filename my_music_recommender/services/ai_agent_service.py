"""
LLM を使って「理由付き推薦文」を作るモジュール。
デフォルトは OpenAI の ChatCompletion (gpt-3.5/4) を想定していますが、
他のモデルに差し替え可能なように抽象化しています。
"""

import os
import openai
import config
openai.api_key = config.OPENAI_API_KEY

def _make_prompt(track_info, similar_tracks):
    """
    トラック情報と類似曲リストを受け取り、LLMに渡すプロンプトを作成。
    """
    lines = []
    lines.append(f"元の曲: \"{track_info.get('name')}\" - {track_info.get('artist')}")
    lines.append("似ている曲の候補:")
    for i, s in enumerate(similar_tracks, 1):
        lines.append(f"{i}. {s.get('name')} - {s.get('artist')} (score: {s.get('score'):.3f})")
    lines.append("")
    lines.append("上の情報をもとに、ユーザーにわかりやすい日本語で「なぜこの曲をおすすめするか」を100〜200文字程度で説明し、各候補を簡単に1行ずつコメントしてください。")
    prompt = "\n".join(lines)
    return prompt

def generate_recommendation_text(track_info, similar_tracks):
    """
    OpenAI に投げてテキストを生成する。
    """
    if not config.OPENAI_API_KEY:
        # フォールバック: 簡易生成
        text = f"おすすめ候補（簡易）: " + ", ".join([f"{t['name']}（{t['artist']}）" for t in similar_tracks])
        return text

    prompt = _make_prompt(track_info, similar_tracks)
    # ChatCompletion を呼ぶ基本例 (gpt-3.5)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"あなたは音楽推薦アシスタントです。"},
            {"role":"user","content":prompt}
        ],
        max_tokens=300,
        temperature=0.8,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # HTMLとして返す場合は適宜整形するが、ここでは生テキスト
    # 返却は safe HTML であることを前提しない（テンプレートで escape されるため）
    return text
