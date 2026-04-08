import os
import ftfy
from tqdm import tqdm

def clean_txt_files(folder_path):
    """
    遍历文件夹，修复 txt 文件中的编码乱码并覆盖原文件
    """
    if not os.path.exists(folder_path):
        print(f"错误: 找不到路径 {folder_path}")
        return

    # 获取所有 txt 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not files:
        print("文件夹中没有 txt 文件。")
        return

    print(f"开始处理 {len(files)} 个文件...")

    for file_name in tqdm(files, desc="清理进度"):
        file_path = os.path.join(folder_path, file_name)

        try:
            # 1. 以 'utf-8' 尝试读取，如果不成功则让 ftfy 处理
            # 即使读取成功，内容中也可能包含 'â\x80\x9c' 这种乱码文本
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 2. 使用 ftfy 修复逻辑乱码
            # fix_text 会自动将 "â€œ" 还原为 "“"
            fixed_content = ftfy.fix_text(content)

            # 3. 如果内容有变化，则写回源文件
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
            
        except Exception as e:
            print(f"\n[错误] 处理文件 {file_name} 时出错: {e}")

if __name__ == "__main__":
    # --- 请修改为你的 txt 文件夹路径 ---
    TARGET_FOLDER = "/root/autodl-tmp/newProject/nontraindata-cleaned" 
    
    # 提醒：操作前建议先备份原始数据
    confirm = input(f"此操作将直接修改 {TARGET_FOLDER} 下的源文件，建议先备份。确定继续吗？(y/n): ")
    if confirm.lower() == 'y':
        clean_txt_files(TARGET_FOLDER)
        print("\n清理完成！乱码已修复并保存至源文件。")
    else:
        print("操作已取消。")