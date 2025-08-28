import requests
import base64
import sys

# ↓↓ ここに、あなたのClient IDを貼り付けてください ↓↓
CLIENT_ID = "c0146d0eab3742b7bbf89c09c0e5588e"

# ↓↓ ここに、あなたのClient Secretを貼り付けてください ↓↓
CLIENT_SECRET = "58c0b49dafaf4757bc9145c6105afd00"

# --- ここから下は編集不要です ---

# サンプルキーのままなら警告を出す
if "c0146d" in CLIENT_ID:
    print("!!! 警告: Client IDがサンプルキーのままです。このテストは失敗する可能性が高いです。")
    print("!!! もしこれがあなた専用のキーで間違いない場合は、このまま続行します。\n")

TOKEN_URL = "https://accounts.spotify.com/api/token"

try:
    print("ステップ1: Spotifyの認証サーバーへの接続を準備します...")
    
    # Client IDとSecretを合体させて、Base64という形式に変換します
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {'grant_type': 'client_credentials'}

    print(f"ステップ2: 認証サーバー ({TOKEN_URL}) にリクエストを送信します...")
    
    # 実際に通信を行う部分
    response = requests.post(TOKEN_URL, headers=headers, data=payload, timeout=10)

    # サーバーからの返事を確認
    response.raise_for_status() # もしエラーなら、ここで例外が発生します

    print("\n★★★ 通信成功！ ★★★")
    print(" -> Spotifyの認証サーバーとの基本的な通信に成功しました。")
    print(" -> サーバーからアクセストークンが返ってきました。")
    print("\n環境やネットワークには問題がない可能性が高いです。")
    # print("受け取ったデータ:", response.json())


except requests.exceptions.HTTPError as e:
    print(f"\n!!! 通信失敗 (HTTPエラー) !!!")
    print(f" -> サーバーからエラーが返されました。")
    print(f" -> ステータスコード: {e.response.status_code}")
    print(f" -> サーバーからのメッセージ: {e.response.text}")
    if e.response.status_code == 403:
        print("\n -> 403 Forbiddenエラーが確認されました。キーが正しい場合、")
        print("    会社のネットワークやセキュリティソフトが通信をブロックしている可能性も考えられます。")

except Exception as e:
    print(f"\n!!! 通信失敗 (予期せぬエラー) !!!")
    print(f" -> Spotifyのサーバーに到達する前に問題が発生した可能性があります。")
    print(f" -> エラー詳細: {e}")
    print("\n -> ファイアウォール、プロキシ、またはインターネット接続を確認してください。")