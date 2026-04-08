import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def run_inference(input_folder, output_folder, base_model_path, lora_path, max_new_tokens=128,min_new_tokens=64):
    # 1. 环境准备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 加载 Tokenizer 和模型
    print(f"正在加载 Tokenizer: {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    # Llama-2 没有默认 pad_token，需要手动指定以避免 batch 推理报错（虽然这里是逐条预测）
    tokenizer.pad_token = tokenizer.eos_token 

    print(f"正在加载模型并挂载 LoRA (这可能需要较多显存)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # 3. 遍历处理 CSV 文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        print(f"\n正在处理文件: {csv_file}")
        
        df = pd.read_csv(input_path)
        prompts = df['prefix'].tolist()
        suffixes = df['suffix'].tolist()
        responses = []

        # 4. 执行推理
        # 使用 torch.no_grad() 禁用梯度计算，节省显存并提速
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="生成进度"):
                # 将文本转为模型输入
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # 生成配置：使用贪婪解码 (do_sample=False)
                output_tokens = model.generate(
                    **inputs,
                    min_new_tokens=max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 贪婪搜索，最能体现记忆化
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 仅获取模型生成的后缀部分（去除掉原始 prompt 占用的 token）
                input_len = inputs.input_ids.shape[1]
                response_tokens = output_tokens[0][input_len:]
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                
                responses.append(response_text)

        # 5. 保存结果
        result_df = pd.DataFrame({
            'prompt': prompts,
            'true_suffix': suffixes,
            'response': responses
        })
        
        output_file_name = csv_file.replace('.csv', '_result.csv')
        result_df.to_csv(os.path.join(output_folder, output_file_name), index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {output_file_name}")

if __name__ == "__main__":
    # --- 路径配置 (根据你的 AutoDL 环境) ---
    INPUT_CSV_DIR = "/root/autodl-tmp/findThreshold/traindata-cleaned/fiction-256token"
    OUTPUT_RESULT_DIR = "/root/autodl-tmp/findThreshold/member_results"
    BASE_MODEL_PATH = "/root/autodl-tmp/llama2-13B"
    LORA_ADAPTER_PATH = "/root/autodl-tmp/final-loss=1.5"
    
    # 设置生成长度，128 token 通常足够判断版权侵权
    run_inference(INPUT_CSV_DIR, OUTPUT_RESULT_DIR, BASE_MODEL_PATH, LORA_ADAPTER_PATH, max_new_tokens=128)