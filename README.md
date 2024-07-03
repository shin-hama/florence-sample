# florence 2 お試し

Microsoft が公開した画像処理用のモデル [Florence 2](https://huggingface.co/microsoft/Florence-2-large) を試す。

## 始め方

メイン PC が Windows だったが、Windows 上で動かすまでのハードルが高かったので nvidia の PyTorch コンテナを利用することにした。

使い始めるには、まず以下コマンドでコンテナを立ち上げる。

```bash
docker compose up -d
```

コンテナが立ち上がったら任意の方法でコンテナに入り、`src\florence_example.py` を実行して動作確認をする。
以下はターミナル上でコンテナに入って実行する方法の一例。

```bash
docker exec -it florence_dev bash
python src/florence_example.py
```
