#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多源数据下载器
支持从 Hugging Face、GitHub、Google Drive 下载数据
"""

import os
import re
import subprocess
import zipfile
from pathlib import Path
import logging
import requests
from huggingface_hub import snapshot_download

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuggingFaceDownloader:
    """
    Hugging Face 数据集下载器
    
    功能：下载指定的 Hugging Face 数据集到本地
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化下载器
        
        参数:
            base_dir (str, optional): 数据存储基础目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, repo_id: str, force_download: bool = False, ignore_patterns: list = None) -> bool:
        """
        下载指定的 Hugging Face 数据集
        
        参数:
            repo_id (str): Hugging Face 仓库ID，例如 'zjunlp/Mol-Instructions'
            force_download (bool): 是否强制重新下载
            ignore_patterns (list): 要忽略的文件模式列表，例如 ['dna/dna_seq.txt', '*.log']
        
        返回:
            bool: 下载成功返回 True
        """
        # 从 repo_id 提取数据集名称作为本地文件夹名
        dataset_name = repo_id.replace('/', '_').replace('-', '_').lower()
        local_dir = self.datasets_dir / dataset_name
        
        # 检查是否已存在
        if not force_download and self._check_exists(local_dir):
            logger.info(f"数据集已存在: {local_dir}")
            return True
        
        try:
            logger.info(f"开始下载数据集: {repo_id}")
            
            # 准备下载参数
            download_kwargs = {
                "repo_id": repo_id,
                "repo_type": "dataset",
                "local_dir": str(local_dir),
                "local_dir_use_symlinks": False
            }
            
            # 如果有忽略模式，添加到参数中
            if ignore_patterns:
                download_kwargs["ignore_patterns"] = ignore_patterns
                logger.info(f"忽略文件模式: {ignore_patterns}")
            
            # 下载数据集
            downloaded_path = snapshot_download(**download_kwargs)
            
            logger.info(f"下载完成: {downloaded_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False
    
    def _check_exists(self, local_dir: Path) -> bool:
        """检查数据集是否已存在"""
        return local_dir.exists() and any(local_dir.iterdir())
    
    def get_dataset_path(self, repo_id: str) -> Path:
        """获取数据集本地路径"""
        dataset_name = repo_id.replace('/', '_').replace('-', '_').lower()
        return self.datasets_dir / dataset_name
    
    def download_with_exclusions(self, repo_id: str, exclude_files: list, force_download: bool = False) -> bool:
        """
        下载数据集但排除指定文件
        
        参数:
            repo_id (str): Hugging Face 仓库ID
            exclude_files (list): 要排除的文件列表，例如 ['dna/dna_seq.txt']
            force_download (bool): 是否强制重新下载
        
        返回:
            bool: 下载成功返回 True
        """
        return self.download(repo_id, force_download, ignore_patterns=exclude_files)


class GitHubDownloader:
    """
    GitHub 仓库下载器
    
    功能：下载指定的 GitHub 仓库到本地
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化下载器
        
        参数:
            base_dir (str, optional): 数据存储基础目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, repo_url: str, force_download: bool = False) -> bool:
        """
        下载指定的 GitHub 仓库
        
        参数:
            repo_url (str): GitHub 仓库URL，例如 'https://github.com/hhnqqq/Biology-Instructions'
            force_download (bool): 是否强制重新下载
        
        返回:
            bool: 下载成功返回 True
        """
        # 从 URL 提取仓库名作为本地文件夹名
        repo_name = repo_url.split('/')[-1].lower().replace('-', '_')
        local_dir = self.datasets_dir / repo_name
        
        # 检查是否已存在
        if not force_download and self._check_exists(local_dir):
            logger.info(f"GitHub 仓库已存在: {local_dir}")
            return True
        
        try:
            logger.info(f"开始下载 GitHub 仓库: {repo_url}")
            
            # 使用 git clone 下载仓库
            cmd = ['git', 'clone', repo_url, str(local_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"下载完成: {local_dir}")
                return True
            else:
                logger.error(f"Git clone 失败: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False
    
    def _check_exists(self, local_dir: Path) -> bool:
        """检查仓库是否已存在"""
        return local_dir.exists() and any(local_dir.iterdir())
    
    def get_dataset_path(self, repo_url: str) -> Path:
        """获取仓库本地路径"""
        repo_name = repo_url.split('/')[-1].lower().replace('-', '_')
        return self.datasets_dir / repo_name


class HttpFileDownloader:
    """
    HTTP 文件下载器
    
    功能：从任意 HTTP/HTTPS 链接下载文件到本地
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化下载器
        
        参数:
            base_dir (str, optional): 数据存储基础目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, download_url: str, force_download: bool = False) -> bool:
        """
        从指定的 HTTP/HTTPS 链接下载文件
        
        参数:
            download_url (str): 文件下载链接
            force_download (bool): 是否强制重新下载
        
        返回:
            bool: 下载成功返回 True
        """
        # 使用统一的路径生成逻辑
        local_dir = self.get_dataset_path(download_url)
        
        # 检查是否已存在
        if not force_download and self._check_exists(local_dir):
            logger.info(f"HTTP 文件已存在: {local_dir}")
            return True
        
        try:
            logger.info(f"开始下载文件: {download_url}")
            
            # 创建本地目录
            local_dir.mkdir(exist_ok=True)
            
            with requests.Session() as session:
                # 设置请求头，模拟浏览器
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = session.get(download_url, headers=headers, stream=True)
                
                if response.status_code == 200:
                    # 确定文件名
                    filename = self._get_filename_from_response(response, download_url)
                    file_path = local_dir / filename
                    
                    logger.info(f"保存文件: {filename}")
                    
                    # 保存文件
                    total_size = 0
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)
                                
                                # 每下载10MB打印一次进度
                                if total_size % (10 * 1024 * 1024) == 0:
                                    logger.info(f"已下载: {total_size / (1024*1024):.1f} MB")
                    
                    logger.info(f"文件下载完成，总大小: {total_size / (1024*1024):.1f} MB")
                    
                    # 如果是zip文件，解压
                    if filename.lower().endswith('.zip'):
                        logger.info("解压ZIP文件...")
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(local_dir)
                        os.remove(file_path)  # 删除zip文件
                        logger.info("ZIP文件解压完成")
                    
                    logger.info(f"下载完成: {local_dir}")
                    return True
                else:
                    logger.error(f"下载失败，状态码: {response.status_code}")
                    return False
            
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return False
    

    
    def _get_filename_from_response(self, response, original_url: str) -> str:
        """从响应头中获取文件名，确保同一链接返回固定文件名"""
        # 首先尝试从Content-Disposition头获取
        content_disposition = response.headers.get('content-disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"').strip("'")
            return filename
        
        # 尝试从原始URL中提取Google Drive文件ID（如果是Google Drive链接）
        url_to_check = original_url
        if hasattr(response, 'url') and response.url:
            url_to_check = response.url
        
        # 检查是否是Google Drive链接，提取文件ID
        if 'drive.google' in url_to_check or 'drive.usercontent.google' in url_to_check:
            file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', url_to_check)
            if file_id_match:
                file_id = file_id_match.group(1)
                return f"stage2_train.jsonl"  # 根据实际情况返回已知的文件名
        
        # 尝试从URL路径中获取文件名（去除动态参数）
        if url_to_check:
            url_parts = url_to_check.split('/')
            if url_parts:
                potential_filename = url_parts[-1].split('?')[0]
                if '.' in potential_filename and len(potential_filename) < 100:  # 避免过长的参数
                    return potential_filename
        
        # 默认文件名使用URL哈希
        clean_url = re.sub(r'[&?](uuid|at|authuser|confirm)=[^&]*', '', original_url)
        url_hash = str(abs(hash(clean_url)))[:10]
        return f"http_file_{url_hash}"
    
    def _check_exists(self, local_dir: Path) -> bool:
        """检查文件是否已存在"""
        return local_dir.exists() and any(local_dir.iterdir())
    
    def get_dataset_path(self, download_url: str) -> Path:
        """获取文件本地路径，确保同一链接返回固定路径"""
        # 检查是否是Google Drive链接，提取文件ID
        if 'drive.google' in download_url or 'drive.usercontent.google' in download_url:
            file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', download_url)
            if file_id_match:
                file_id = file_id_match.group(1)
                return self.datasets_dir / f"gdrive_{file_id}"
        
        # 对于其他链接，使用URL哈希（去除动态参数后计算）
        # 去除常见的动态参数
        clean_url = re.sub(r'[&?](uuid|at|authuser|confirm)=[^&]*', '', download_url)
        url_hash = str(abs(hash(clean_url)))[:10]
        return self.datasets_dir / f"http_{url_hash}"


def main():
    """
    主函数 - 下载所有类型的数据集
    """
    print("🚀 开始下载多源数据集...")
    
    # Hugging Face 数据集配置 (repo_id, exclude_files)
    hf_datasets = [
        ('zjunlp/Mol-Instructions', None),
        ('PharMolix/UniProtQA', None), 
        ('EMCarrami/Pika-DS', None),
        ('InstaDeepAI/ChatNT_training_data', None),
        ('dnagpt/llama-gene-train-data', ['dna/dna_seq.txt', 'protein/protein_seq.txt'])  # 示例：排除大文件
    ]
    
    # GitHub 仓库
    github_repos = [
        'https://github.com/hhnqqq/Biology-Instructions'
    ]
    
    # HTTP 文件下载链接
    http_files = [
        'https://drive.usercontent.google.com/download?id=1OC3VpPKSQ0VHd9ZeZhnxI8EA2wTdrBg5&export=download&authuser=0&confirm=t&uuid=e785cd67-94d7-46b6-a1b4-14f643624b7d&at=AN8xHor66TqLKD-p6ddJcODmObWc%3A1756722563687'
    ]
    
    total_success = 0
    total_count = len(hf_datasets) + len(github_repos) + len(http_files)
    
    # 下载 Hugging Face 数据集
    print("\n📥 下载 Hugging Face 数据集...")
    hf_downloader = HuggingFaceDownloader()
    for i, (repo_id, exclude_files) in enumerate(hf_datasets, 1):
        print(f"[{i}/{len(hf_datasets)}] {repo_id}")
        if exclude_files:
            print(f"    排除文件: {exclude_files}")
        
        if hf_downloader.download(repo_id, ignore_patterns=exclude_files):
            print(f"✅ 成功")
            total_success += 1
        else:
            print(f"❌ 失败")
    
    # 下载 GitHub 仓库
    print("\n📥 下载 GitHub 仓库...")
    gh_downloader = GitHubDownloader()
    for i, repo_url in enumerate(github_repos, 1):
        print(f"[{i}/{len(github_repos)}] {repo_url}")
        if gh_downloader.download(repo_url):
            print(f"✅ 成功")
            total_success += 1
        else:
            print(f"❌ 失败")
    
    # 下载 HTTP 文件
    print("\n📥 下载 HTTP 文件...")
    http_downloader = HttpFileDownloader()
    for i, download_url in enumerate(http_files, 1):
        print(f"[{i}/{len(http_files)}] {download_url[:80]}...")
        if http_downloader.download(download_url):
            print(f"✅ 成功")
            total_success += 1
        else:
            print(f"❌ 失败")
    
    print(f"\n🎉 完成！成功下载 {total_success}/{total_count} 个数据集")
    print(f"📁 存储位置: {hf_downloader.datasets_dir}")


if __name__ == "__main__":
    main()