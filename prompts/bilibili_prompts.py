from typing import Any, Dict, List
from prompts import YA_MCPServer_Prompt

# 导入分类器
try:
    from modules.YA_Common.utils.video_classifier import get_classifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False


@YA_MCPServer_Prompt(
    name="bilibili_today_hot_with_analysis",
    title="Bilibili Today Hot with AI Analysis",
    description="今日B站热门视频及AI分类分析报告",
)
async def bilibili_today_hot_analysis() -> Dict[str, Any]:
    """生成今日B站热门视频的AI分析提示。
    
    Returns:
        Dict[str, Any]: 包含分析报告和推荐。
    """
    # 这里可以调用tool获取数据，或假设数据已传入
    # 简化版本：生成分析框架提示
    
    prompt_text = """📊 今日B站热门视频AI分析报告

🎯 内容分类分布：
根据AI智能分类，今日热门视频主要分布如下：
- {等待数据填充}

🔥 热门标签云：
{等待数据填充}

💡 观看建议：
基于您的兴趣偏好，推荐关注以下类型内容：
1. {等待数据填充}
2. {等待数据填充}
3. {等待数据填充}

📈 趋势洞察：
{等待数据填充}

需要我获取具体的今日热门数据并进行详细分析吗？"""
    
    return {
        "prompt": prompt_text,
        "context": {
            "type": "bilibili_trend_analysis",
            "requires_data": True,
            "classification_method": "jieba_tfidf" if CLASSIFIER_AVAILABLE else "none",
        }
    }


@YA_MCPServer_Prompt(
    name="personalized_bilibili_recommend",
    title="Personalized Bilibili Recommend",
    description="基于用户兴趣的个性化B站视频推荐",
)
async def personalized_recommend(
    user_interests: str,
    preferred_categories: str = "",
    mood: str = "relax",
) -> Dict[str, Any]:
    """基于用户兴趣和AI分类生成个性化推荐。
    
    Args:
        user_interests (str): 用户兴趣描述，如"喜欢科技、游戏、动漫"。
        preferred_categories (str, optional): 偏好的视频分类，逗号分隔。
        mood (str, optional): 当前心情，如"relax", "excited", "learning", "bored"。
        
    Returns:
        Dict[str, Any]: 个性化推荐提示。
    """
    # 解析用户兴趣
    interests = [i.strip() for i in user_interests.split(",") if i.strip()]
    
    # 心情映射到分类权重
    mood_boost = {
        "relax": ["生活", "美食", "音乐", "动画"],
        "excited": ["游戏", "娱乐", "运动", "挑战"],
        "learning": ["知识", "科技", "历史", "纪录片"],
        "bored": ["搞笑", "娱乐", "鬼畜", "挑战"],
        "sad": ["治愈", "音乐", "萌宠", "生活"],
    }.get(mood, [])
    
    # 构建推荐策略
    target_cats = list(set(preferred_categories.split(",")) & set(mood_boost)) if preferred_categories else mood_boost
    
    prompt_text = f"""🎯 为您定制的B站观看指南

基于您的兴趣【{', '.join(interests)}】和当前【{mood}】心情：

📌 推荐关注分类：
{chr(10).join(f'• {cat}' for cat in target_cats[:5])}

🔍 建议搜索关键词：
{_generate_search_keywords(interests, target_cats)}

📊 今日热门匹配：
我可以为您筛选今日热门视频中符合以上偏好的内容，需要吗？

💡 智能提醒：
当【{', '.join(target_cats[:2])}】类内容在热门榜占比超过30%时，通常是该类内容质量较高的时期，建议重点关注！"""
    
    return {
        "prompt": prompt_text,
        "context": {
            "user_interests": interests,
            "mood": mood,
            "target_categories": target_cats,
            "recommendation_strategy": "ai_personalized",
        }
    }


def _generate_search_keywords(interests: List[str], categories: List[str]) -> str:
    """生成搜索关键词建议"""
    keyword_map = {
        "科技": "AI 数码评测 编程教程",
        "游戏": "Steam新游 电竞赛事 游戏解说",
        "美食": "探店打卡 家常菜教程 美食测评",
        "生活": "Vlog日常 RoomTour 自律打卡",
        "知识": "科普纪录片 读书分享 技能教程",
        "动画": "新番导视 动漫解说 手书MAD",
    }
    
    keywords = []
    for cat in categories[:3]:
        keywords.append(keyword_map.get(cat, cat))
    
    return " | ".join(keywords)