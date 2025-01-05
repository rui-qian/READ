from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder
import os

# 设置模型名称和用户名
model_name = "READ-LLaVA-v1.5-7B-for-ReasonSeg-valset"
username = "rui-qian"  # 替换为你的 Hugging Face 用户名
repo_url = f"https://huggingface.co/{username}/{model_name}"

# 登录 Hugging Face
from huggingface_hub import login
login()  # 运行时会提示输入 Hugging Face API Token

# 创建模型仓库
try:
    create_repo(repo_id=model_name, private=False)  # private=True 可设置为私有模型
    print(f"仓库 {model_name} 创建成功！")
except Exception as e:
    print(f"仓库创建失败: {e}")

# 上传文件夹
folder_path = "READ-LLaVA-v1.5-7B-for-ReasonSeg-valset"  # 替换为你的模型文件保存目录
if os.path.exists(folder_path):
    try:
        upload_folder(
            folder_path=folder_path,  # 本地文件夹路径
            repo_id=f"{username}/{model_name}",  # 仓库名称
            commit_message="初次上传",  # 提交信息
        )
        print(f"文件夹 {folder_path} 上传成功！")
    except Exception as e:
        print(f"文件夹上传失败: {e}")
else:
    print(f"文件夹 {folder_path} 不存在，请检查路径。")
