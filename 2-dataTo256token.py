import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def process_txt_with_lora(input_folder, output_folder, base_model_path, num_samples=100, prefix_len=256, suffix_len=128):
    # 1. 加载 Tokenizer
    print(f"正在加载 Tokenizer: {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 总共需要的长度
    total_required_len = prefix_len + suffix_len

    # 3. 遍历处理文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    for file_name in tqdm(files, desc="处理进度"):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将全文转为 token IDs
        tokens = tokenizer.encode(content, add_special_tokens=False)
        total_tokens = len(tokens)
        
        # 检查长度是否足以支撑一个采样周期
        if total_tokens < total_required_len:
            print(f"\n[跳过] {file_name} 长度不足 {total_required_len} tokens (当前: {total_tokens})")
            continue
        
        # 4. 等步长采样逻辑
        # 这里的 max_start_idx 必须为 prefix 的起始点留出后面 1024 token 的空间
        max_start_idx = total_tokens - total_required_len
        
        sampled_prefixes = []
        sampled_suffixes = []
        
        for i in range(num_samples):
            # 计算当前 prefix 的起始索引
            if num_samples > 1:
                start_idx = int(i * max_start_idx / (num_samples - 1))
            else:
                start_idx = 0
            
            # 截取 prefix (256 tokens)
            prefix_tokens = tokens[start_idx : start_idx + prefix_len]
            prefix_text = tokenizer.decode(prefix_tokens, skip_special_tokens=True)
            
            # 紧接着截取 suffix (1024 tokens)
            suffix_tokens = tokens[start_idx + prefix_len : start_idx + total_required_len]
            suffix_text = tokenizer.decode(suffix_tokens, skip_special_tokens=True)
            
            sampled_prefixes.append(prefix_text)
            sampled_suffixes.append(suffix_text)
        
        # 5. 保存至 CSV
        df = pd.DataFrame({
            'prefix': sampled_prefixes,
            'suffix': sampled_suffixes
        })
        
        output_file_name = file_name.replace('.txt', '.csv')
        df.to_csv(os.path.join(output_folder, output_file_name), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # --- 路径配置 ---
    INPUT_DIR = "/root/autodl-tmp/findThreshold/traindata-cleaned/fiction"
    OUTPUT_DIR = "/root/autodl-tmp/findThreshold/traindata-cleaned/fiction-256token"
    BASE_MODEL_PATH = "/root/autodl-tmp/llama2-13B"
    
    # 执行处理：100个样本，256前缀，128后缀
    process_txt_with_lora(
        INPUT_DIR, 
        OUTPUT_DIR, 
        BASE_MODEL_PATH, 
        num_samples=100, 
        prefix_len=256, 
        suffix_len=128
    )
    print("\n数据切分完成，CSV 已包含 prefix 和 suffix。")