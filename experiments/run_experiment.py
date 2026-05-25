"""
实验主脚本：对比三种 RAG Pipeline 在 HotpotQA 上的表现。

用法：
  python experiments/run_experiment.py --pipeline naive --size 100
  python experiments/run_experiment.py --pipeline self  --size 100
  python experiments/run_experiment.py --pipeline all   --size 100

结果保存到 results/tables/ 目录。
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from index.build_index import load_index
from pipelines.naive_rag import NaiveRAGPipeline
from pipelines.self_rag import SelfReflectiveRAGPipeline
from pipelines.agentic_rag import AgenticRAGPipeline
from eval.metrics import evaluate_batch
from config import DATA_DIR, RESULTS_DIR


def load_dataset(data_path: str, size: int = None):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if size and size < len(data):
        data = data[:size]
    return data


def run_pipeline(pipeline, dataset: list, desc: str = "") -> dict:
    """在整个数据集上运行一个 Pipeline，返回结果字典。"""
    predictions, ground_truths = [], []
    raw_results = []
    api_calls_list = []
    latency_list = []

    pipeline.reset_stats()

    for item in tqdm(dataset, desc=desc):
        question = item["question"]
        gold_answer = item["answer"]

        t0 = time.time()
        result = pipeline.answer(question)
        latency = time.time() - t0

        predictions.append(result["answer"])
        ground_truths.append(gold_answer)
        api_calls_list.append(result["api_calls"])
        latency_list.append(latency)

        raw_results.append({
            "id":           item["id"],
            "question":     question,
            "gold_answer":  gold_answer,
            "pred_answer":  result["answer"],
            "type":         item["type"],
            "api_calls":    result["api_calls"],
            "latency_s":    round(latency, 3),
            "steps":        result.get("steps", []),
        })

    metrics = evaluate_batch(predictions, ground_truths)

    # 按问题类型分层统计
    bridge_preds = [r["pred_answer"] for r in raw_results if r["type"] == "bridge"]
    bridge_golds = [r["gold_answer"] for r in raw_results if r["type"] == "bridge"]
    comp_preds   = [r["pred_answer"] for r in raw_results if r["type"] == "comparison"]
    comp_golds   = [r["gold_answer"] for r in raw_results if r["type"] == "comparison"]

    bridge_metrics = evaluate_batch(bridge_preds, bridge_golds) if bridge_preds else {}
    comp_metrics   = evaluate_batch(comp_preds, comp_golds)     if comp_preds   else {}

    summary = {
        "pipeline":        pipeline.name,
        "n_samples":       len(dataset),
        "overall_em":      round(metrics["em"],  4),
        "overall_f1":      round(metrics["f1"],  4),
        "bridge_em":       round(bridge_metrics.get("em", 0), 4),
        "bridge_f1":       round(bridge_metrics.get("f1", 0), 4),
        "comparison_em":   round(comp_metrics.get("em", 0), 4),
        "comparison_f1":   round(comp_metrics.get("f1", 0), 4),
        "avg_api_calls":   round(sum(api_calls_list) / len(api_calls_list), 2),
        "total_api_calls": sum(api_calls_list),
        "total_tokens":    pipeline.total_tokens,
        "est_cost_usd":    round(pipeline.cost_estimate_usd(), 4),
        "avg_latency_s":   round(sum(latency_list) / len(latency_list), 2),
    }

    return {"summary": summary, "raw": raw_results}


def save_results(results: dict, pipeline_name: str, timestamp: str):
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "raw"), exist_ok=True)

    # 保存汇总表格（JSON）
    summary_path = os.path.join(RESULTS_DIR, "tables",
                                f"{pipeline_name}_{timestamp}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, indent=2)

    # 保存逐题原始结果
    raw_path = os.path.join(RESULTS_DIR, "raw",
                            f"{pipeline_name}_{timestamp}_raw.json")
    with open(raw_path, "w") as f:
        json.dump(results["raw"], f, ensure_ascii=False, indent=2)

    print(f"  保存到: {summary_path}")
    return summary_path


def print_summary(summary: dict):
    print(f"\n{'='*55}")
    print(f"  Pipeline: {summary['pipeline']}")
    print(f"  样本数:   {summary['n_samples']}")
    print(f"{'─'*55}")
    print(f"  Overall  EM: {summary['overall_em']:.4f}   F1: {summary['overall_f1']:.4f}")
    print(f"  Bridge   EM: {summary['bridge_em']:.4f}   F1: {summary['bridge_f1']:.4f}")
    print(f"  Comparsn EM: {summary['comparison_em']:.4f}   F1: {summary['comparison_f1']:.4f}")
    print(f"{'─'*55}")
    print(f"  平均 API 调用次数: {summary['avg_api_calls']:.2f}")
    print(f"  总 token 消耗:     {summary['total_tokens']:,}")
    print(f"  预估费用 (USD):    ${summary['est_cost_usd']:.4f}")
    print(f"  平均延迟:          {summary['avg_latency_s']:.2f}s/题")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="AgentRAG-Eval 实验脚本")
    parser.add_argument("--pipeline", choices=["naive", "self", "agentic", "all"],
                        default="naive", help="选择运行哪个 Pipeline")
    parser.add_argument("--size",     type=int, default=None,
                        help="使用数据集前 N 条（默认全部）")
    parser.add_argument("--data",     type=str,
                        default=os.path.join(DATA_DIR, "hotpotqa_100.json"))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"加载数据集: {args.data}")
    dataset = load_dataset(args.data, size=args.size)
    print(f"  使用 {len(dataset)} 条数据")

    print("加载向量索引...")
    collection, _ = load_index()
    print(f"  索引大小: {collection.count()} 个段落")

    pipelines_to_run = {
        "naive":   NaiveRAGPipeline,
        "self":    SelfReflectiveRAGPipeline,
        "agentic": AgenticRAGPipeline,
    }

    if args.pipeline != "all":
        pipelines_to_run = {args.pipeline: pipelines_to_run[args.pipeline]}

    all_summaries = []
    for name, PipelineClass in pipelines_to_run.items():
        print(f"\n运行 {name.upper()} RAG Pipeline...")
        pipeline = PipelineClass(collection)
        results = run_pipeline(pipeline, dataset, desc=f"  {name}")
        print_summary(results["summary"])
        save_results(results, name, timestamp)
        all_summaries.append(results["summary"])

    # 如果运行了多个 Pipeline，打印对比表格
    if len(all_summaries) > 1:
        print(f"\n{'='*65}")
        print(f"  对比汇总")
        print(f"{'─'*65}")
        print(f"  {'Pipeline':<20} {'EM':>7} {'F1':>7} {'API调用':>8} {'费用USD':>9}")
        print(f"{'─'*65}")
        for s in all_summaries:
            print(f"  {s['pipeline']:<20} {s['overall_em']:>7.4f} {s['overall_f1']:>7.4f}"
                  f" {s['avg_api_calls']:>8.2f} {s['est_cost_usd']:>9.4f}")
        print(f"{'='*65}")

        # 保存对比表
        compare_path = os.path.join(RESULTS_DIR, "tables",
                                    f"comparison_{timestamp}.json")
        with open(compare_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n对比结果已保存: {compare_path}")


if __name__ == "__main__":
    main()
