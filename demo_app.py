from __future__ import annotations

import argparse
from functools import lru_cache
from html import escape

import gradio as gr

from report_rag.config import DEFAULT_LLM_MODEL, DEFAULT_RERANKER_MODEL
from report_rag.search import (
    SearchSession,
    generation_context_results,
    llm_exclusion_reasons,
)


def make_args(
    index_dir: str,
    route: str,
    vector_weight: float,
    top_k: int,
    max_chars: int,
    rerank: bool,
    rerank_top_n: int,
    generate: bool,
    context_chunks: int,
    min_recall_score: float,
    min_rerank_score: float,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    no_citation_check: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        index_dir=index_dir,
        route=route,
        vector_weight=vector_weight,
        top_k=int(top_k),
        max_chars=int(max_chars),
        ivf_nprobe=16,
        hnsw_ef_search=64,
        json=False,
        no_fp16=False,
        rerank=rerank,
        reranker_model=DEFAULT_RERANKER_MODEL,
        rerank_top_n=int(rerank_top_n),
        rerank_batch_size=8,
        generate=generate,
        llm_model=DEFAULT_LLM_MODEL,
        context_chunks=int(context_chunks),
        max_input_tokens=int(max_input_tokens),
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        min_recall_score=min_recall_score,
        min_rerank_score=min_rerank_score,
        citation_check=not no_citation_check,
    )


@lru_cache(maxsize=2)
def get_session(
    index_dir: str,
    route: str,
    rerank: bool,
    generate: bool,
) -> SearchSession:
    args = make_args(
        index_dir=index_dir,
        route=route,
        vector_weight=0.65,
        top_k=5,
        max_chars=800,
        rerank=rerank,
        rerank_top_n=20,
        generate=generate,
        context_chunks=5,
        min_recall_score=0.3,
        min_rerank_score=-2.0,
        max_input_tokens=6000,
        max_new_tokens=512,
        temperature=0.0,
        no_citation_check=False,
    )
    return SearchSession(args)


def render_retrieved_chunks(
    results: list[dict],
    args: argparse.Namespace,
) -> str:
    context_results = generation_context_results(results, args) if args.generate else []
    fed_ids = {result["chunk_id"] for result in context_results}
    parts = []

    for result in results:
        fed = result["chunk_id"] in fed_ids
        if fed:
            status = "Fed into the LLM"
            reason = ""
            color = "#e8f7ee"
        elif args.generate:
            reasons = llm_exclusion_reasons(result, args)
            reason = "; ".join(reasons) if reasons else "context_chunks limit reached"
            status = "Excluded from LLM"
            color = "#fff1f0"
        else:
            reason = "generation disabled"
            status = "Retrieved only"
            color = "#f5f5f5"

        rerank_score = result["rerank_score"]
        score_line = (
            f"score={result['score']:.4f} | recall={result['recall_score']:.4f} | "
            f"vector={result['vector_score']:.4f} | bm25={result['bm25_score']:.4f} | "
            f"rerank={rerank_score}"
        )
        reason_html = f"<p><b>Reason:</b> {escape(reason)}</p>" if reason else ""
        text = escape(result["text"][: args.max_chars])
        parts.append(
            f'''
            <details open style="background:{color}; border:1px solid #ddd;
                    border-radius:8px; padding:10px; margin:10px 0;">
              <summary><b>[{result['rank']}] {status}</b> - {escape(result['source'])}
              pages:{escape(str(result['pages']))}</summary>
              {reason_html}
              <p><b>{escape(score_line)}</b></p>
              <p><b>chunk_id:</b> {escape(result['chunk_id'])}</p>
              <pre style="white-space:pre-wrap; font-family:inherit;">{text}</pre>
            </details>
            '''
        )

    return "\n".join(parts) if parts else "<p>No retrieved chunks.</p>"


def answer_query(
    query: str,
    index_dir: str,
    route: str,
    vector_weight: float,
    top_k: int,
    max_chars: int,
    rerank: bool,
    rerank_top_n: int,
    generate: bool,
    context_chunks: int,
    min_recall_score: float,
    min_rerank_score: float,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    no_citation_check: bool,
) -> tuple[str, str]:
    query = query.strip()
    if not query:
        return "请输入 query。", ""

    args = make_args(
        index_dir=index_dir,
        route=route,
        vector_weight=vector_weight,
        top_k=top_k,
        max_chars=max_chars,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        generate=generate,
        context_chunks=context_chunks,
        min_recall_score=min_recall_score,
        min_rerank_score=min_rerank_score,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        no_citation_check=no_citation_check,
    )

    session = get_session(index_dir, route, rerank, generate)
    session.args = args
    results, answer = session.search(query)

    answer_markdown = answer if answer is not None else "未启用生成，仅显示召回结果。"
    chunks_html = render_retrieved_chunks(results, args)
    return answer_markdown, chunks_html


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Report RAG Demo") as demo:
        gr.Markdown("# Report RAG Demo\n输入问题，查看召回文档、LLM 回答和引用来源。")
        with gr.Row():
            query = gr.Textbox(
                label="Query",
                placeholder="例如：介绍一下L4 级自动驾驶",
                lines=3,
                scale=4,
            )
            submit = gr.Button("Search / Generate", variant="primary", scale=1)

        with gr.Accordion("参数", open=False):
            with gr.Row():
                index_dir = gr.Textbox(label="Index Dir", value="index_1024")
                route = gr.Dropdown(
                    label="Route",
                    choices=["hybrid", "vector", "bm25"],
                    value="hybrid",
                )
                vector_weight = gr.Slider(
                    label="Vector Weight",
                    minimum=0,
                    maximum=1,
                    value=0.65,
                    step=0.05,
                )
            with gr.Row():
                top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
                max_chars = gr.Slider(
                    label="Max Chars per Chunk",
                    minimum=200,
                    maximum=3000,
                    value=800,
                    step=100,
                )
                context_chunks = gr.Slider(
                    label="Context Chunks",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                )
            with gr.Row():
                rerank = gr.Checkbox(label="Rerank", value=True)
                rerank_top_n = gr.Slider(
                    label="Rerank Top N",
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                )
                generate = gr.Checkbox(label="Generate Answer", value=True)
                no_citation_check = gr.Checkbox(label="Disable Citation Check", value=False)
            with gr.Row():
                min_recall_score = gr.Number(label="Min Recall Score", value=0.3)
                min_rerank_score = gr.Number(label="Min Rerank Score", value=-2.0)
            with gr.Row():
                max_input_tokens = gr.Slider(
                    label="Max Input Tokens",
                    minimum=512,
                    maximum=16000,
                    value=6000,
                    step=512,
                )
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=64,
                    maximum=2048,
                    value=512,
                    step=64,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.05,
                )

        answer = gr.Markdown(label="最终回答 + 引用来源")
        chunks = gr.HTML(label="召回文档")

        inputs = [
            query,
            index_dir,
            route,
            vector_weight,
            top_k,
            max_chars,
            rerank,
            rerank_top_n,
            generate,
            context_chunks,
            min_recall_score,
            min_rerank_score,
            max_input_tokens,
            max_new_tokens,
            temperature,
            no_citation_check,
        ]
        submit.click(answer_query, inputs=inputs, outputs=[answer, chunks])
        query.submit(answer_query, inputs=inputs, outputs=[answer, chunks])

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=7860)
