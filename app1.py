import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from patent_pipeline import (
    analyze_claim,
    check_ollama,
    OLLAMA_MODEL
)


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="日本語特許請求項SAO構造分析",
    page_icon="🪼",
    layout="wide"
)


# =========================================================
# タイトル
# =========================================================

st.title(
    "🪼 日本語特許請求項SAO構造分析"
)

st.caption(
    "ローカルLLM＋GiNZAによる階層SAO解析"
)


# =========================================================
# Ollama状態
# =========================================================

if check_ollama():

    st.success(
        f"🟢 Ollama接続OK：{OLLAMA_MODEL}"
    )

else:

    st.error(
        "🔴 Ollamaに接続できません。"
    )

    st.info(
        "Ollamaを起動してから解析してください。"
    )


# =========================================================
# 入力
# =========================================================

default_claim = """放熱装置と、
前記放熱装置の主面に配置された少なくとも１つの取り付けフレームと、
スイッチング機能を有する少なくとも１つのパワー半導体モジュールと、
を備え、
前記パワー半導体モジュールは、正側電源入力端子、負側電源入力端子および出力端子を含み、
前記取り付けフレームは、少なくとも１つの開口部を有し、
前記パワー半導体モジュールは、前記開口部により前記取り付けフレームに対して位置決めされており、
前記取り付けフレームの一部は、前記正側電源入力端子、前記負側電源入力端子および前記出力端子と、前記放熱装置との間に位置する、
インテリジェントパワーモジュール。"""


claim = st.text_area(
    "特許請求項",
    value=default_claim,
    height=330
)


# =========================================================
# 解析
# =========================================================

if st.button(
    "🔍 SAO解析を実行",
    type="primary",
    use_container_width=True
):

    if not claim.strip():

        st.warning(
            "特許請求項を入力してください。"
        )

        st.stop()

    if not check_ollama():

        st.error(
            "Ollamaが起動していません。"
        )

        st.stop()

    try:

        with st.spinner(
            "ローカルLLMでSAO解析中..."
        ):

            result = analyze_claim(
                claim
            )

        st.session_state[
            "analysis_result"
        ] = result

        st.success(
            "SAO解析が完了しました。"
        )

    except Exception as e:

        st.error(
            "解析中にエラーが発生しました。"
        )

        st.exception(e)

        st.stop()


# =========================================================
# 結果
# =========================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]

    sao = result["sao"]

    tabs = st.tabs([
        "🌳 階層SAO",
        "🔗 SAO表",
        "🧩 GiNZA",
        "🕸️ グラフ",
        "📄 JSON"
    ])


    # =====================================================
    # 階層
    # =====================================================

    with tabs[0]:

        st.subheader(
            "🌳 階層SAO構造"
        )

        node_map = {
            node["id"]: node
            for node in sao["nodes"]
        }

        relations = sao["relations"]

        children = {}

        for relation in relations:

            children.setdefault(
                relation["subject_id"],
                []
            ).append(
                relation
            )


        def display_node(
            node_id,
            depth=0
        ):

            node = node_map.get(
                node_id
            )

            if node is None:
                return

            st.markdown(
                f'{"　" * depth}'
                f'**{node["text"]}**'
            )

            for relation in children.get(
                node_id,
                []
            ):

                target = node_map.get(
                    relation["object_id"]
                )

                if target is None:
                    continue

                st.markdown(
                    f'{"　" * (depth + 1)}'
                    f'└─ `{relation["action"]}` → '
                    f'**{target["text"]}**'
                )

                display_node(
                    target["id"],
                    depth + 2
                )


        object_ids = {
            r["object_id"]
            for r in relations
        }

        roots = [
            node
            for node in sao["nodes"]
            if node["id"] not in object_ids
        ]

        for root in roots:

            display_node(
                root["id"]
            )


    # =====================================================
    # SAO表
    # =====================================================

    with tabs[1]:

        st.subheader(
            "🔗 SAOトリプル"
        )

        node_map = {
            node["id"]: node["text"]
            for node in sao["nodes"]
        }

        rows = []

        for relation in sao[
            "relations"
        ]:

            rows.append({

                "Subject":
                    node_map.get(
                        relation[
                            "subject_id"
                        ],
                        ""
                    ),

                "Action":
                    relation[
                        "action"
                    ],

                "Object":
                    node_map.get(
                        relation[
                            "object_id"
                        ],
                        ""
                    ),

                "階層":
                    relation[
                        "level"
                    ],

                "関係":
                    relation[
                        "relation_type"
                    ]
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # GiNZA
    # =====================================================

    with tabs[2]:

        st.subheader(
            "🧩 GiNZA係り受け解析"
        )

        df = pd.DataFrame(
            result["ginza"]["tokens"]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # グラフ
    # =====================================================

    with tabs[3]:

        st.subheader(
            "🕸️ SAOネットワーク"
        )

        G = nx.DiGraph()

        for node in sao["nodes"]:

            G.add_node(
                node["text"]
            )

        for relation in sao[
            "relations"
        ]:

            subject = node_map.get(
                relation[
                    "subject_id"
                ]
            )

            object_ = node_map.get(
                relation[
                    "object_id"
                ]
            )

            if subject and object_:

                G.add_edge(
                    subject,
                    object_,
                    action=relation[
                        "action"
                    ]
                )

        if G.number_of_nodes():

            fig, ax = plt.subplots(
                figsize=(16, 10)
            )

            pos = nx.spring_layout(
                G,
                seed=42,
                k=2
            )

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=2500,
                ax=ax
            )

            nx.draw_networkx_edges(
                G,
                pos,
                arrows=True,
                arrowsize=20,
                ax=ax
            )

            nx.draw_networkx_labels(
                G,
                pos,
                font_size=9,
                ax=ax
            )

            labels = nx.get_edge_attributes(
                G,
                "action"
            )

            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=labels,
                font_size=8,
                ax=ax
            )

            ax.axis("off")

            st.pyplot(
                fig,
                use_container_width=True
            )


    # =====================================================
    # JSON
    # =====================================================

    with tabs[4]:

        st.subheader(
            "📄 SAO JSON"
        )

        st.json(
            sao
        )
