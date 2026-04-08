import os
import pandas as pd
import numpy as np
import Levenshtein
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm

# --- 1. 相似度计算函数 ---

def calc_edit_similarity(true_text, pred_text):
    """计算归一化编辑相似度 (0-1)，值越大越相似"""
    if not true_text and not pred_text: return 1.0
    if not true_text or not pred_text: return 0.0
    dist = Levenshtein.distance(true_text, pred_text)
    max_len = max(len(true_text), len(pred_text))
    return 1.0 - (dist / max_len)

def calc_rouge_l(scorer, true_text, pred_text):
    """计算 ROUGE-L F1 分数 (0-1)"""
    if not true_text or not pred_text: return 0.0
    scores = scorer.score(true_text, pred_text)
    return scores['rougeL'].fmeasure

# --- 2. 核心评估流程 ---

def evaluate_results(member_dir, non_member_dir, output_file):
    print("正在加载 Embedding 模型 (用于余弦相似度)...")
    # 使用一个轻量级但效果很好的多语言模型
    embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def process_folder(folder_path, label):
        """读取文件夹下所有结果并计算相似度分数"""
        all_data = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        for f in tqdm(files, desc=f"处理数据 (Label={label})"):
            df = pd.read_csv(os.path.join(folder_path, f))
            # 填补可能出现的 NaN（模型生成为空的情况）
            df['true_suffix'] = df['true_suffix'].fillna("")
            df['response'] = df['response'].fillna("")
            
            true_texts = df['true_suffix'].tolist()
            pred_texts = df['response'].tolist()
            
            # 计算字面级别指标
            edit_sims = [calc_edit_similarity(t, p) for t, p in zip(true_texts, pred_texts)]
            rouge_ls = [calc_rouge_l(scorer, t, p) for t, p in zip(true_texts, pred_texts)]
            
            # 计算语义级别指标 (Embedding Cosine Similarity)
            embeddings_true = embed_model.encode(true_texts, convert_to_tensor=True, show_progress_bar=False)
            embeddings_pred = embed_model.encode(pred_texts, convert_to_tensor=True, show_progress_bar=False)
            cosine_sims = util.cos_sim(embeddings_true, embeddings_pred).diagonal().cpu().numpy().tolist()
            
            for i in range(len(df)):
                all_data.append({
                    'label': label,
                    'edit_sim': edit_sims[i],
                    'rouge_l': rouge_ls[i],
                    'cosine_sim': cosine_sims[i]
                })
        return pd.DataFrame(all_data)

    print("\n--- 开始处理成员数据集 (侵权) ---")
    df_member = process_folder(member_dir, label=1)
    
    print("\n--- 开始处理非成员数据集 (无辜) ---")
    df_non_member = process_folder(non_member_dir, label=0)
    
    # 合并数据
    df_all = pd.concat([df_member, df_non_member], ignore_index=True)
    
    # 构建第四个指标：综合得分 (三种分数的平均值)
    df_all['combined_score'] = (df_all['edit_sim'] + df_all['rouge_l'] + df_all['cosine_sim']) / 3.0
    
    # 导出打分结果供后续画图使用
    df_all.to_csv(output_file, index=False)
    print(f"\n所有相似度打分已保存至: {output_file}")

    # --- 3. 统计指标与寻找最佳阈值 ---
    print("\n" + "="*50)
    print(" 综合评估报告 (寻找最佳阈值)")
    print("="*50)

    metrics = ['edit_sim', 'rouge_l', 'cosine_sim', 'combined_score']
    metric_names = ['编辑距离相似度', 'ROUGE-L 分数', 'Embedding余弦相似度', '三个指标联合(均值)']

    for metric_col, metric_name in zip(metrics, metric_names):
        y_true = df_all['label'].values
        y_scores = df_all[metric_col].values

        # 1. 寻找最佳 F1 对应的阈值
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
        # 避免除以 0 的警告
        f1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 1.0
        best_f1 = f1_scores[best_f1_idx]
        best_p = precision[best_f1_idx]
        best_r = recall[best_f1_idx]

        # 2. 计算低假阳性 (FPR <= 1%) 下的真阳性率 (TPR)
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        # 找到满足 FPR <= 0.01 的最大索引
        low_fpr_mask = fpr <= 0.01 
        tpr_at_low_fpr = tpr[low_fpr_mask][-1]
        threshold_at_low_fpr = thresholds_roc[low_fpr_mask][-1]

        print(f"\n【{metric_name}】")
        print(f"  --> 基于最高 F1-Score 确定的 [最佳阈值]: {best_f1_threshold:.4f}")
        print(f"      - 该阈值下的表现: Precision = {best_p:.4f}, Recall = {best_r:.4f}, F1 = {best_f1:.4f}")
        print(f"  --> 安全合规标准 (FPR极低): 当阈值提升至 {threshold_at_low_fpr:.4f} 时 (仅允许 1% 误报)")
        print(f"      - 此时能抓出的侵权比例 (TPR@1%FPR) = {tpr_at_low_fpr:.4f}")

if __name__ == "__main__":
    # --- 路径配置 ---
    # 请将成员数据的 _result.csv 放入以下文件夹
    MEMBER_RESULTS_DIR = "/root/autodl-tmp/results_member" 
    # 请将非成员数据的 _result.csv 放入以下文件夹
    NON_MEMBER_RESULTS_DIR = "/root/autodl-tmp/results_non_member" 
    
    # 所有打分数据的输出文件
    OUTPUT_SCORE_CSV = "all_evaluation_scores.csv"

    evaluate_results(MEMBER_RESULTS_DIR, NON_MEMBER_RESULTS_DIR, OUTPUT_SCORE_CSV)