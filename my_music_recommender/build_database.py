# build_database.py (診断モード版)
import os
import sys
import numpy as np
import json

print("--- データベース構築スクリプト（診断モード）を開始します ---")

# --- 設定 ---
# SpotifyのプレイリストID (例: "37i9dQZEVXbKXQ4mDTEBXq" は "Top 50 - Global")
TARGET_PLAYLIST_ID = "37i9dQZEVXbKXQ4mDTEBXq"
# 処理する曲数（最大100）
TRACK_LIMIT = 50
# -------------

try:
    print("[1/7] 必要なモジュールをインポートしています...")
    # tqdmをインポートしようと試みる
    try:
        from tqdm import tqdm
    except ImportError:
        print(" -> 'tqdm'ライブラリが見つかりません。インストールします...")
        # pipを使ってtqdmをインストール
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
        print(" -> 'tqdm'のインストールが完了しました。")

    from services.spotify_service import SpotifyService
    from services.vggish_service import extract_audio_vector
    import config
    print(" -> モジュールのインポートに成功しました。")
except ImportError as e:
    print(f"!!! エラー: 必要なモジュールが見つかりません。({e})")
    print("!!! このスクリプトは 'my_music_recommender' ディレクトリ直下で実行してください。")
    sys.exit(1)

def build():
    """
    Spotifyのプレイリストから楽曲を取得し、ベクトルデータベースを構築する
    """
    try:
        print("[2/7] Spotify Serviceを初期化しています...")
        spotify = SpotifyService()
        print(" -> Spotify Serviceの初期化に成功しました。")
    except Exception as e:
        print(f"!!! エラー: Spotify Serviceの初期化に失敗しました: {e}")
        print("!!! config.pyのSPOTIPY_CLIENT_IDとSPOTIPY_CLIENT_SECRETが正しく設定されているか確認してください。")
        return

    print(f"[3/7] プレイリスト '{TARGET_PLAYLIST_ID}' から最大 {TRACK_LIMIT} 曲の情報を取得します...")
    try:
        playlist = spotify.sp.playlist_items(TARGET_PLAYLIST_ID, limit=TRACK_LIMIT)
        track_items = playlist.get("items", [])
        if not track_items:
            print("!!! 警告: プレイリストから曲を取得できませんでした。IDが正しいか、公開されているか確認してください。")
            return
        print(f" -> {len(track_items)} 曲の情報を取得しました。")
    except Exception as e:
        print(f"!!! エラー: Spotify APIからのプレイリスト取得に失敗しました: {e}")
        return

    all_vectors = []
    all_meta = []

    print(f"[4/7] 取得した {len(track_items)} 曲のベクトルを生成します...")
    for item in tqdm(track_items, desc="楽曲処理中"):
        track = item.get("track")
        if not track or not track.get("id") or not track.get("preview_url"):
            tqdm.write(f"スキップ: {track.get('name', '不明な曲')} (IDまたはプレビュー音源なし)")
            continue

        track_id = track["id"]
        track_name = track.get("name", "不明な曲")

        try:
            track_info = spotify.get_track_info(track_id)
            spotify_vec = spotify.features_to_vector(track_info.get("features", {}), dim=config.VECTOR_DIM)
            audio_source = track_info.get("preview_url")
            vggish_vec = extract_audio_vector(audio_source, dim=config.VGGISH_FEATURE_DIM)
            combined_vec = np.concatenate([spotify_vec, vggish_vec])

            all_vectors.append(combined_vec)
            all_meta.append({"id": track_info["id"], "name": track_info["name"], "artist": track_info["artist"]})
        except Exception as e:
            tqdm.write(f"エラー: \"{track_name}\" の処理中にエラーが発生。スキップします。 (詳細: {e})")

    if not all_vectors:
        print("\\n!!! 警告: 有効なベクトルが1つも生成されませんでした。処理を終了します。")
        return

    print(f"\\n[5/7] データベースファイルをNumpy配列に変換しています...")
    vectors_np = np.array(all_vectors, dtype=np.float32)
    print(" -> 変換に成功しました。")

    print(f"[6/7] データベースファイルを保存しています...")
    print(f" -> ベクトルファイル (形状: {vectors_np.shape}) を {config.COMBINED_VECTORS_NPY} に保存します。")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    np.save(config.COMBINED_VECTORS_NPY, vectors_np)

    print(f" -> メタ情報 ({len(all_meta)}件) を {config.VECTORS_META} に保存します。")
    with open(config.VECTORS_META, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(" -> 保存に成功しました。")

    print("\\n[7/7] データベースの構築が完了しました！")


if __name__ == "__main__":
    try:
        if os.path.exists(config.COMBINED_VECTORS_NPY):
            res = input(f"既存のDBファイル '{config.COMBINED_VECTORS_NPY}' が見つかりました。上書きしますか？ (y/N): ").lower()
            if res != 'y':
                print("処理を中断しました。")
                sys.exit(0)
        build()
    except Exception as e:
        print(f"\\n!!! 予期せぬエラーでスクリプトが停止しました: {e}")
        import traceback
        traceback.print_exc()