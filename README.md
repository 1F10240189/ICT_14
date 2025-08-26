このディレクトリにはトラックのメタやベクトルが入ります。

- sample_tracks.json : サンプルのトラックメタ（audio_path がローカル音声のパス）
- track_vectors.npy : ベクトルDB の numpy 配列（N x D）
- track_meta.json : ベクトルに対応するメタ情報（JSON配列）

最初は存在しなくてもアプリはデモ用のランダムDBを作成します。実運用では:
1) 各トラックの音源を揃える（ローカルファイル or preview_url のダウンロード）
2) services/vggish_service.extract_audio_vector を使ってベクトルを作る
3) vector_db.add(...) でDBに追加する
