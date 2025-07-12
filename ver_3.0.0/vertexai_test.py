from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="635939494644",
      location="us-central1",
  )

  msg1_text1 = types.Part.from_text(text="""あなたは麻雀の専門コーチです。AI分析結果に基づいて、打牌判断の戦術的根拠を分かりやすく説明してください。 【局面状況】 局: 東2局 (南家) リーチ者: 0人 残り牌: 60枚 ドラ: 7p 供託: 0本 / 本場: 0本場 【自分の手牌】 6m 7m 9m 9m 3p 4p 4p 8p 8p 9p 3s 6s 7s 西 (P2) 【各プレイヤーの捨て牌】 P0: 2s 1s P1: 北 南 白 P2: 9s 北 ← 自分 P3: 南 北 【AI判断】 推奨打牌: 西 (確信度: 82.2%) 実際打牌: 9p 【推奨打牌Top5】 1位: 西 (82.2%) 2位: 9p (3.1%) ★実打牌 3位: 4p (3.0%) 4位: 9m (1.1%) 5位: 3p (0.8%) 【受け入れ分析比較】 ■ AI推奨打牌 打西 → 3向聴 受け入れ: 06789m2340789p30678s (計55枚) 詳細: 0m×4 6m×3 7m×3 8m×4 9m×2 2p×4 3p×3 4p×2 0p×4 7p×4 8p×2 9p×3 3s×3 0s×4 6s×3 7s×3 8s×4 ■ 実際打牌 打9p → 3向聴 受け入れ: 06789m23408p30678s3z (計51枚) 詳細: 0m×4 6m×3 7m×3 8m×4 9m×2 2p×4 3p×3 4p×2 0p×4 8p×2 3s×3 0s×4 6s×3 7s×3 8s×4 3z×3 ■ 比較結果 シャンテン数: 同じ (3向聴) 受け入れ枚数: AI推奨が有利 (55枚 vs 51枚) 【AI思考プロセス】 ■ 手牌評価 ・西が最重要要素(重要度0.564) → 不要牌として強く推奨 ■ 注目した相手の動き（層別分析） 【Layer 1】 1. プレイヤー1が白を捨てた (注目度: 0.0739) 2. プレイヤー3が北を捨てた (注目度: 0.0736) 3. 自分のツモ牌を考慮 (注目度: 0.0736) 4. プレイヤー2が北を捨てた (注目度: 0.0735) 5. プレイヤー1が北を捨てた (注目度: 0.0731) 6. プレイヤー1が南を捨てた (注目度: 0.0729) 7. プレイヤー3が南を捨てた (注目度: 0.0729) 8. プレイヤー2が9sを捨てた (注目度: 0.0725) 【Layer 2】 1. 初期配牌の影響 (注目度: 0.0767) 2. 自分のツモ牌を考慮 (注目度: 0.0744) 3. プレイヤー3が北を捨てた (注目度: 0.0725) 4. プレイヤー1が白を捨てた (注目度: 0.0724) 5. プレイヤー2が北を捨てた (注目度: 0.0718) 6. プレイヤー1が北を捨てた (注目度: 0.0714) 7. プレイヤー0が1sを捨てた (注目度: 0.0713) 8. 自分のツモ牌を考慮 (注目度: 0.0709) 【Layer 3】 1. 初期配牌の影響 (注目度: 0.0795) 2. 自分のツモ牌を考慮 (注目度: 0.0780) 3. プレイヤー1が北を捨てた (注目度: 0.0751) 4. プレイヤー2が9sを捨てた (注目度: 0.0745) 5. プレイヤー3が南を捨てた (注目度: 0.0728) 6. ドラ表示の影響 (注目度: 0.0704) 7. プレイヤー3が北を捨てた (注目度: 0.0700) 8. 自分のツモ牌を考慮 (注目度: 0.0699) 【Layer 4】 1. 初期配牌の影響 (注目度: 0.0790) 2. プレイヤー2が9sを捨てた (注目度: 0.0790) 3. プレイヤー1が北を捨てた (注目度: 0.0788) 4. 自分のツモ牌を考慮 (注目度: 0.0766) 5. プレイヤー0が1sを捨てた (注目度: 0.0731) 6. プレイヤー1が白を捨てた (注目度: 0.0709) 7. 自分のツモ牌を考慮 (注目度: 0.0708) 8. プレイヤー3が南を捨てた (注目度: 0.0708) ■ 戦略方針: バランス型 ・安全性と速度の両方を考慮した判断 【解説要求】 以下の3つの観点から、初心者にも分かりやすく解説してください： 1. **即効性判断** (50文字以内) なぜこの牌を切るのが良いのか、端的な理由 2. **戦術的根拠** (150文字以内) 手牌構成、局面状況を踏まえた詳細な戦術理由 3. **代替案検討** (100文字以内) 他の選択肢と比べてなぜこれがベストか 各項目を明確に分けて、実戦で使える知識として説明してください。""")

  model = "projects/635939494644/locations/us-central1/endpoints/429703437786021888"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()