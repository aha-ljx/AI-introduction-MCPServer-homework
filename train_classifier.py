# train_classifier.py
import sys
import os
import requests  # 需要requests
import time
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.YA_Common.utils.video_classifier import VideoClassifier


def _sync_fetch_bilibili_hot(limit: int) -> dict:
    """
    分批获取B站热门视频，避免触发反爬
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bilibili.com",
        "Accept": "application/json, text/plain, */*",
    }
    
    session = requests.Session()
    session.get("https://www.bilibili.com", headers=headers, timeout=10)
    
    all_videos = []
    batch_size = 20  # 每批20条
    pages = (limit + batch_size - 1) // batch_size  # 计算需要多少页
    
    for page in range(1, pages + 1):
        # 计算当前页需要多少条
        current_ps = min(batch_size, limit - len(all_videos))
        
        print(f"获取第 {page}/{pages} 页，请求 {current_ps} 条...")
        
        response = session.get(
            "https://api.bilibili.com/x/web-interface/popular",
            params={
                "ps": current_ps,      # 每页数量
                "pn": page,             # 页码
            },
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get("code") != 0:
            print(f"第 {page} 页错误: {data.get('message')}")
            break
        
        videos = data.get("data", {}).get("list", [])
        all_videos.extend(videos)
        print(f"  获取到 {len(videos)} 条，累计 {len(all_videos)} 条")
        
        # 不是最后一页，添加延迟
        if page < pages:
            delay = random.uniform(0.5, 1.5)  # 随机延迟0.5-1.5秒
            print(f"  等待 {delay:.2f} 秒...")
            time.sleep(delay)
    
    # 构造统一返回格式
    return {
        "code": 0,
        "message": "OK",
        "data": {
            "list": all_videos[:limit]  # 确保不超过限制
        }
    }


# ========== 后续代码不变 ==========
CATEGORY_MAP = {
    "动画": "动画", "番剧": "动画", "国创": "动画", "鬼畜": "动画",
    "音乐": "音乐", "舞蹈": "音乐", "演奏": "音乐",
    "游戏": "游戏", "电子竞技": "游戏", "单机游戏": "游戏",
    "科技": "科技", "数码": "科技", "计算机技术": "科技",
    "知识": "科技", "科学科普": "科技", "社科人文": "科技",
    "生活": "生活", "日常": "生活", "VLOG": "生活",
    "美食": "美食", "美食制作": "美食",
    "娱乐": "娱乐", "综艺": "娱乐", "搞笑": "娱乐",
    "影视": "娱乐", "电影": "娱乐", "纪录片": "娱乐",
    "运动": "其他", "健身": "其他",
    "汽车": "其他",
    "时尚": "其他", "美妆": "其他",
}


def collect_training_data_from_bilibili(sample_size: int = 300) -> list:
    """从B站热门自动采集训练数据"""
    print(f"正在从B站热门采集 {sample_size} 条训练数据...")
    
    try:
        print("调用 _sync_fetch_bilibili_hot...")
        data = _sync_fetch_bilibili_hot(sample_size)
        print(f"API返回 code: {data.get('code')}")
        print(f"API返回 message: {data.get('message')}")
        
    except Exception as e:
        print(f"采集失败: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    if data.get("code") != 0:
        print(f"B站API错误: {data.get('message')}")
        return []
    
    videos = data.get("data", {}).get("list", [])
    print(f"成功获取 {len(videos)} 条视频")
    
    training_data = []
    category_stats = {}
    
    for v in videos:
        title = v.get("title", "").strip()
        desc = v.get("desc", "").strip()
        tname = v.get("tname", "").strip()
        
        label = CATEGORY_MAP.get(tname, "其他")
        
        if not title or not tname:
            continue
            
        training_data.append((title, desc, tname, label))
        category_stats[label] = category_stats.get(label, 0) + 1
    
    print("\n分类分布统计：")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}条")
    
    return training_data


def prepare_training_data(training_data: list) -> tuple:
    """预处理训练数据"""
    print(f"\n预处理 {len(training_data)} 条训练数据...")
    
    temp_classifier = VideoClassifier(model_path=None)
    
    texts = []
    labels = []
    skipped = 0
    
    for title, desc, tname, label in training_data:
        if not title or len(title) < 3:
            skipped += 1
            continue
            
        processed = temp_classifier.preprocess_text(title, desc, tname)
        
        if len(processed) < 5:
            skipped += 1
            continue
        
        texts.append(processed)
        labels.append(label)
    
    print(f"有效样本: {len(texts)}, 跳过: {skipped}")
    
    unique_pairs = list(set(zip(texts, labels)))
    if unique_pairs:
        texts, labels = zip(*unique_pairs)
        texts, labels = list(texts), list(labels)
    else:
        texts, labels = [], []
    
    print(f"去重后样本: {len(texts)}")
    
    return texts, labels


def main():
    """主训练流程"""
    training_data = collect_training_data_from_bilibili(sample_size=300)
    
    if len(training_data) < 100:
        print("错误：采集到的样本太少，无法训练。")
        return
    
    texts, labels = prepare_training_data(training_data)
    
    if len(texts) < 50:
        print("错误：有效样本不足50条，无法训练。")
        return
    
    print("\n开始训练集成模型...")
    classifier = VideoClassifier(model_path=None)
    
    result = classifier.train(
        texts=list(texts), 
        labels=list(labels), 
        save_path="ensemble_model.pkl"
    )
    
    print("\n" + "="*50)
    print("训练完成！")
    print(f"模型保存路径: {os.path.abspath('ensemble_model.pkl')}")
    print(f"训练集准确率: {result['train_accuracy']:.4f}")
    print(f"验证集准确率: {result['val_accuracy']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()