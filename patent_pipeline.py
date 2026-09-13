import json
import re
from typing import Dict, Any

import requests
import spacy


# =========================================================
# 設定
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

# 最初に使うモデル
OLLAMA_MODEL = "gpt-oss:20b"


# =========================================================
# GiNZA
# =========================================================

_NLP = None


def get_nlp():

    global _NLP

    if _NLP is None:

        try:
            _NLP = spacy.load("ja_ginza")

        except Exception:

            try:
                _NLP = spacy.load("ja_ginza_electra")

            except Exception as e:

                raise RuntimeError(
                    "GiNZAモデルを読み込めませんでした。"
                ) from e

    return _NLP


# =========================================================
# 前処理
# =========================================================

def preprocess_claim(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    text = text.replace("\u3000", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"^\s*【?請求項\s*[0-9０-９]+\s*】?\s*",
        "",
        text
    )

    return text.strip()


# =========================================================
# GiNZA解析
# =========================================================

def ginza_parse(
    text: str
) -> Dict[str, Any]:

    nlp = get_nlp()

    doc = nlp(text)

    tokens = []

    for token in doc:

        tokens.append({
            "id": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.i,
            "head_text": token.head.text
        })

    sentences = []

    for sent in doc.sents:

        sentences.append({
            "text": sent.text,
            "start": sent.start,
            "end": sent.end
        })

    return {
        "text": text,
        "tokens": tokens,
        "sentences": sentences
    }


# =========================================================
# SAO JSON Schema
# =========================================================

SAO_SCHEMA = {

    "type": "object",

    "properties": {

        "claim_subject": {
            "type": "string"
        },

        "nodes": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "id": {
                        "type": "string"
                    },

                    "text": {
                        "type": "string"
                    },

                    "node_type": {
                        "type": "string"
                    },

                    "parent_id": {
                        "type": [
                            "string",
                            "null"
                        ]
                    }
                },

                "required": [
                    "id",
                    "text",
                    "node_type",
                    "parent_id"
                ]
            }
        },

        "relations": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "subject_id": {
                        "type": "string"
                    },

                    "action": {
                        "type": "string"
                    },

                    "object_id": {
                        "type": "string"
                    },

                    "relation_type": {
                        "type": "string"
                    },

                    "level": {
                        "type": "integer"
                    }
                },

                "required": [
                    "subject_id",
                    "action",
                    "object_id",
                    "relation_type",
                    "level"
                ]
            }
        }
    },

    "required": [
        "claim_subject",
        "nodes",
        "relations"
    ]
}


# =========================================================
# LLMプロンプト
# =========================================================

SYSTEM_PROMPT = r"""
あなたは日本語特許請求項のSAO構造解析システムです。

入力された特許請求項から、

Subject
Action
Object

のSAO関係を抽出してください。

さらに、SAOを階層構造として表現してください。

==================================================
重要ルール
==================================================

【1. 勝手に関係を作らない】

本文に存在しない関係を推測して追加してはいけません。

==================================================

【2. 前記】

「前記」は新しいノードにしません。

例えば、

パワー半導体モジュール
前記パワー半導体モジュール

は同じノードとして扱います。

==================================================

【3. 数量表現】

「少なくとも」
「少なくとも１つ」

などは通常ノードにしません。

==================================================

【4. 備える】

請求項の最上位構造として扱います。

例えば、

「インテリジェントパワーモジュールは、
放熱装置と、取り付けフレームと、
パワー半導体モジュールとを備える」

なら、

インテリジェントパワーモジュール
 └─ 備える
     ├─ 放熱装置
     ├─ 取り付けフレーム
     └─ パワー半導体モジュール

です。

==================================================

【5. 有する】

「有する」を「備える」に置換してはいけません。

例えば、

取り付けフレームは開口部を有する

なら、

取り付けフレーム
 └─ 有する
     └─ 開口部

です。

==================================================

【6. 含む】

「含む」も独立した関係として扱います。

例えば、

パワー半導体モジュールは
正側電源入力端子、負側電源入力端子
および出力端子を含む

なら、

パワー半導体モジュール
 └─ 含む
     ├─ 正側電源入力端子
     ├─ 負側電源入力端子
     └─ 出力端子

です。

==================================================

【7. 機能】

「スイッチング機能を有する
パワー半導体モジュール」

なら、

パワー半導体モジュール
 └─ 有する
     └─ スイッチング機能

です。

==================================================

【8. 位置関係】

例えば、

「取り付けフレームの一部は、
端子と放熱装置との間に位置する」

なら、

Subject:
取り付けフレームの一部

Action:
位置する

Object:
端子と放熱装置との間

です。

絶対に、

放熱装置 → 位置する → インテリジェントパワーモジュール

などと逆転させないでください。

==================================================

【9. Aの一部】

「取り付けフレームの一部」

は必要に応じて独立したノードとして扱います。

==================================================

【10. 並列】

「A、BおよびC」

のような並列構造は、

A
B
C

を別々のノードとして扱います。

==================================================

【11. 動詞】

「含み」→「含む」
「有し」→「有する」
「位置決めされている」→「位置決めされる」

のように基本形にします。

==================================================

【12. 階層】

level 0:
請求項

level 1:
主要構成

level 2:
構成要素内部

level 3:
さらに下位

としてください。

==================================================

【13. parent_id】

各ノードの直接の親を指定してください。

例えば、

インテリジェントパワーモジュール
    ↓
パワー半導体モジュール
    ↓
出力端子

なら、

パワー半導体モジュール.parent_id
=
インテリジェントパワーモジュールのID

出力端子.parent_id
=
パワー半導体モジュールのID

です。

==================================================

【14. 技術用語】

「放熱装置」
「パワー半導体モジュール」
「正側電源入力端子」

などの技術的名詞句は、できるだけ原文を保持してください。

==================================================

【15. 最重要】

SAOの意味関係を正確にすることを最優先してください。

"""


# =========================================================
# Ollama接続確認
# =========================================================

def check_ollama():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        if response.status_code == 200:
            return True

    except Exception:
        pass

    return False


# =========================================================
# LLM SAO抽出
# =========================================================

def extract_sao_with_llm(
    claim_text: str,
    ginza_result: Dict[str, Any]
):

    if not check_ollama():

        raise RuntimeError(
            "Ollamaに接続できません。\n\n"
            "PowerShellで「ollama list」を実行して、"
            "Ollamaがインストールされているか確認してください。"
        )

    ginza_text = json.dumps(
        ginza_result,
        ensure_ascii=False,
        indent=2
    )

    user_prompt = f"""
以下の日本語特許請求項を解析してください。

【特許請求項】

{claim_text}


【GiNZA解析結果】

{ginza_text}


上記を参考にして、指定されたJSON Schemaに従って
階層SAOを抽出してください。

GiNZAの解析が意味的に誤っている場合は、
特許請求項本文を優先してください。
"""

    payload = {

        "model": OLLAMA_MODEL,

        "messages": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        "stream": False,

        "format": SAO_SCHEMA,

        "options": {
            "temperature": 0
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Ollamaに接続できません。\n\n"
            "Ollamaが起動していることを確認してください。"
        )

    if response.status_code != 200:

        raise RuntimeError(
            "Ollama APIエラー:\n"
            + response.text
        )

    data = response.json()

    content = data[
        "message"
    ][
        "content"
    ]

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        raise RuntimeError(
            "LLMが正しいJSONを返しませんでした。\n\n"
            + content
        )


# =========================================================
# SAO検証
# =========================================================

def validate_sao(
    result: Dict[str, Any]
):

    node_ids = {
        node["id"]
        for node in result.get(
            "nodes",
            []
        )
    }

    valid_relations = []

    for relation in result.get(
        "relations",
        []
    ):

        subject = relation[
            "subject_id"
        ]

        object_ = relation[
            "object_id"
        ]

        if subject not in node_ids:
            continue

        if object_ not in node_ids:
            continue

        if subject == object_:
            continue

        valid_relations.append(
            relation
        )

    result["relations"] = valid_relations

    return result


# =========================================================
# グラフデータ
# =========================================================

def build_graph_data(
    sao_result
):

    nodes = sao_result.get(
        "nodes",
        []
    )

    relations = sao_result.get(
        "relations",
        []
    )

    node_map = {
        node["id"]: node
        for node in nodes
    }

    edges = []

    for relation in relations:

        subject = node_map.get(
            relation["subject_id"]
        )

        object_ = node_map.get(
            relation["object_id"]
        )

        if subject is None:
            continue

        if object_ is None:
            continue

        edges.append({

            "source": subject["text"],

            "action": relation[
                "action"
            ],

            "target": object_[
                "text"
            ],

            "level": relation[
                "level"
            ],

            "relation_type":
                relation[
                    "relation_type"
                ]
        })

    return {

        "nodes": nodes,

        "edges": edges
    }


# =========================================================
# 全体解析
# =========================================================

def analyze_claim(
    claim_text: str
):

    processed = preprocess_claim(
        claim_text
    )

    ginza_result = ginza_parse(
        processed
    )

    sao_result = extract_sao_with_llm(
        processed,
        ginza_result
    )

    sao_result = validate_sao(
        sao_result
    )

    graph_data = build_graph_data(
        sao_result
    )

    return {

        "original_text":
            claim_text,

        "processed_text":
            processed,

        "ginza":
            ginza_result,

        "sao":
            sao_result,

        "graph":
            graph_data
    }
