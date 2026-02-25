import httpx
from tools import YA_MCPServer_Tool
from typing import Any, Dict, List, Optional
import asyncio
import requests
from datetime import datetime
# ========== 新增：导入分类器 ==========
try:
    from modules.YA_Common.utils.video_classifier import get_classifier, VideoClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    print("[Warning] VideoClassifier not available, classification disabled")

try:
    from modules.YA_Common.utils.report_generator import ReportGenerator
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False
    print("[Warning] ReportGenerator not available")

def _sync_fetch_bilibili_hot(limit: int) -> Dict[str, Any]:
    """
    同步获取B站热门视频
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bilibili.com",
        "Accept": "application/json, text/plain, */*",
    }
    
    session = requests.Session()
    session.get("https://www.bilibili.com", headers=headers, timeout=10)
    
    response = session.get(
        "https://api.bilibili.com/x/web-interface/popular",
        params={"ps": limit, "pn": 1},
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    
    return response.json()


def _sync_fetch_bilibili_video(bvid: str) -> Dict[str, Any]:
    """
    同步获取B站视频详情
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bilibili.com",
        "Accept": "application/json, text/plain, */*",
    }
    
    session = requests.Session()
    session.get("https://www.bilibili.com", headers=headers, timeout=10)
    
    response = session.get(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    
    return response.json()

def _classify_video_sync(title: str, desc: str, tname: str) -> Optional[Dict[str, Any]]:
    """同步执行视频分类"""
    if not CLASSIFIER_AVAILABLE:
        return None
    
    classifier = get_classifier()
    classification = classifier.predict(title, desc, tname)
    tags = classifier.extract_tags(title, desc, num_tags=5)
    
    return {
        "classification": classification,
        "tags": tags
    }

def _generate_insights(trend: Dict, videos: List[Dict]) -> List[str]:
    """生成趋势洞察"""
    insights = []
    
    # 主导分类洞察
    dominant = trend.get("dominant_category")
    if dominant:
        pct = trend["category_distribution"][dominant]["percentage"]
        insights.append(f"今日热门以【{dominant}】类内容为主导，占比{pct}%")
    
    # 多样性洞察
    diversity = trend.get("diversity", 0)
    if diversity > 0.5:
        insights.append("内容类型丰富多样，涵盖多个领域")
    else:
        insights.append("内容类型相对集中，话题较为单一")
    
    # 热门标签洞察
    top_tags = trend.get("top_tags", [])
    if top_tags:
        tag_names = [t["tag"] for t in top_tags[:3]]
        insights.append(f"热门话题包括：{', '.join(tag_names)}")
    
    return insights


def _get_top_videos_by_category(videos: List[Dict], top_n: int = 3) -> Dict[str, List[Dict]]:
    """按分类获取热门视频"""
    result = {}
    
    for cat in set(v.get("classification", {}).get("primary", "未知") for v in videos):
        cat_videos = [
            {
                "title": v["title"][:30] + "..." if len(v["title"]) > 30 else v["title"],
                "bvid": v["bvid"],
                "play_count": v["play_count"],
            }
            for v in videos
            if v.get("classification", {}).get("primary") == cat
        ]
        # 按播放量排序
        cat_videos.sort(key=lambda x: x["play_count"], reverse=True)
        result[cat] = cat_videos[:top_n]
    
    return result


@YA_MCPServer_Tool(
    name="get_bilibili_hot",
    title="Get Bilibili Hot Videos",
    description="获取哔哩哔哩当前热门视频列表（带AI智能分类）",
)
async def get_bilibili_hot(
    limit: int = 10,
    generate_report: bool = False,  # 新增参数
) -> Dict[str, Any]:
    """获取哔哩哔哩当前热门视频列表，并使用AI进行内容分类和标签提取。
    
    Args:
        limit (int, optional): 返回的视频数量，默认为10，最大为50。
        generate_report (bool, optional): 是否生成可视化报告，默认为False。
        
    Returns:
        Dict[str, Any]: 包含热门视频列表及AI分类标签的字典。
    """
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(f"无法导入httpx模块: {e}")

    limit = min(max(limit, 1), 50)

    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _sync_fetch_bilibili_hot, limit)
        
        if data.get("code") != 0:
            raise RuntimeError(f"B站API错误: {data.get('message', 'Unknown error')}")
        
        videos = data.get("data", {}).get("list", [])
        
        formatted_videos: List[Dict[str, Any]] = []
        for video in videos:
            bvid = video.get("bvid", "")
            title = video.get("title", "")
            desc = video.get("desc", "")
            tname = video.get("tname", "")  # B站原始分区
            
            # ========== 新增：AI分类和标签提取 ==========
            ai_result = None
            if CLASSIFIER_AVAILABLE:
                ai_result = await loop.run_in_executor(
                    None,
                    _classify_video_sync,
                    title,
                    desc,
                    tname
                )
            
            video_data = {
                "title": title,
                "bvid": bvid,
                "author": video.get("owner", {}).get("name", ""),
                "author_mid": video.get("owner", {}).get("mid", 0),
                "play_count": video.get("stat", {}).get("view", 0),
                "like_count": video.get("stat", {}).get("like", 0),
                "coin_count": video.get("stat", {}).get("coin", 0),
                "favorite_count": video.get("stat", {}).get("favorite", 0),
                "share_count": video.get("stat", {}).get("share", 0),
                "comment_count": video.get("stat", {}).get("reply", 0),
                "duration": video.get("duration", 0),
                "pic": video.get("pic", "").strip(),
                "url": f"https://www.bilibili.com/video/{bvid}",  # 修正：移除空格
                "bilibili_category": tname,  # B站原始分类
            }
            
            # 添加AI分类结果
            if ai_result:
                video_data["ai_classification"] = ai_result["classification"]
                video_data["ai_tags"] = ai_result["tags"]
            else:
                video_data["ai_classification"] = None
                video_data["ai_tags"] = []
            
            formatted_videos.append(video_data)
        
        # ========== 新增：生成趋势分析报告 ==========
        trend_analysis = None
        if CLASSIFIER_AVAILABLE and formatted_videos:
            classifier = get_classifier()
            trend_analysis = await loop.run_in_executor(
                None,
                classifier.analyze_trend,
                formatted_videos
            )
        
        # ========== 构建返回结果 ==========
        result = {
            "source": "bilibili",
            "category": "hot_videos",
            "total": len(formatted_videos),
            "videos": formatted_videos,
            "ai_analysis": {
                "classifier_available": CLASSIFIER_AVAILABLE,
                "trend_summary": trend_analysis,
                "classification_method": "ensemble" if CLASSIFIER_AVAILABLE else "none",
            } if trend_analysis else None
        }
        
        # ========== 生成可视化报告 ==========
        if generate_report and REPORT_AVAILABLE and formatted_videos:
            try:
                generator = ReportGenerator(output_dir="reports")
                
                # 构造趋势数据
                trend_data = {
                    "trend_analysis": trend_analysis or {},
                    "videos": formatted_videos,
                }
                
                report_files = await loop.run_in_executor(
                    None,
                    generator.generate_all_reports,
                    trend_data,
                    formatted_videos
                )
                
                # 添加到返回结果
                result["report_files"] = report_files
                
            except Exception as e:
                result["report_error"] = str(e)
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"获取B站热门视频失败: {e}")


# ========== 新增：专门的分类工具 ==========
@YA_MCPServer_Tool(
    name="classify_video",
    title="Classify Video Content",
    description="对单个视频进行AI内容分类和标签提取",
)
async def classify_video(
    title: str,
    description: str = "",
    bilibili_category: str = "",
) -> Dict[str, Any]:
    """对视频内容进行AI分类分析。
    
    Args:
        title (str): 视频标题。
        description (str, optional): 视频描述。
        bilibili_category (str, optional): B站原始分类。
        
    Returns:
        Dict[str, Any]: 分类结果和提取的标签。
    """
    if not CLASSIFIER_AVAILABLE:
        return {
            "error": "分类器未可用，请检查modules.YA_Common.utils.video_classifier模块",
            "title": title,
        }
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _classify_video_sync,
            title,
            description,
            bilibili_category
        )
        
        if result is None:
            return {"error": "分类失败"}
        
        return {
            "input": {
                "title": title,
                "description": description[:200] if description else "",
                "bilibili_category": bilibili_category,
            },
            "classification": result["classification"],
            "extracted_tags": result["tags"],
            "suggested_categories": [
                c["category"] for c in result["classification"].get("top3", [])
            ],
        }
        
    except Exception as e:
        return {"error": f"分类过程出错: {str(e)}"}



@YA_MCPServer_Tool(
    name="get_bilibili_video_info",
    title="Get Bilibili Video Info",
    description="根据BV号获取哔哩哔哩视频详细信息",
)
async def get_bilibili_video_info(bvid: str) -> Dict[str, Any]:
    """根据BV号获取哔哩哔哩视频详细信息。

    Args:
        bvid (str): 视频的BV号（例如 "BV1xx411c7mD"）。

    Returns:
        Dict[str, Any]: 包含视频详细信息的字典。
    """
    try:
        import httpx
    except ImportError as e:
        raise RuntimeError(f"无法导入httpx模块: {e}")

    try:
        # Windows下使用线程池执行同步请求
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _sync_fetch_bilibili_video, bvid)
        
        if data.get("code") != 0:
            raise RuntimeError(f"B站API错误: {data.get('message', 'Unknown error')}")
        
        video = data.get("data", {})
        owner = video.get("owner", {})
        stat = video.get("stat", {})
        
        return {
            "source": "bilibili",
            "bvid": bvid,
            "title": video.get("title", ""),
            "description": video.get("desc", ""),
            "author": owner.get("name", ""),
            "author_mid": owner.get("mid", 0),
            "play_count": stat.get("view", 0),
            "like_count": stat.get("like", 0),
            "coin_count": stat.get("coin", 0),
            "favorite_count": stat.get("favorite", 0),
            "share_count": stat.get("share", 0),
            "comment_count": stat.get("reply", 0),
            "duration": video.get("duration", 0),
            "pic": video.get("pic", "").strip(),
            "url": f"https://www.bilibili.com/video/{bvid}"
        }
        
    except Exception as e:
        raise RuntimeError(f"获取B站视频信息失败: {e}")
    
# ========== 新增：批量分类工具 ==========
@YA_MCPServer_Tool(
    name="analyze_todays_trend",
    title="Analyze Today's Bilibili Trend",
    description="分析今日B站热门视频的分类趋势和特征，生成可视化报告",
)
async def analyze_todays_trend(
    sample_size: int = 50,
    generate_report: bool = True,  # 新增参数
) -> Dict[str, Any]:
    """获取今日热门视频并进行全面的趋势分析。
    
    Args:
        sample_size: 采样视频数量，默认50，最大100
        generate_report: 是否生成可视化报告，默认True
        
    Returns:
        趋势分析报告，包含报告文件路径
    """
    if not CLASSIFIER_AVAILABLE:
        return {"error": "分类器未可用"}
    
    sample_size = min(max(sample_size, 10), 100)
    
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _sync_fetch_bilibili_hot, sample_size)
        
        if data.get("code") != 0:
            return {"error": "获取B站数据失败"}
        
        videos = data.get("data", {}).get("list", [])
        
        # 批量分类
        classified_videos = []
        for v in videos:
            result = await loop.run_in_executor(
                None,
                _classify_video_sync,
                v.get("title", ""),
                v.get("desc", ""),
                v.get("tname", "")
            )
            
            # 格式化视频数据
            video_data = {
                "title": v.get("title", ""),
                "bvid": v.get("bvid", ""),
                "author": v.get("owner", {}).get("name", ""),
                "pic": v.get("pic", ""),
                "play_count": v.get("stat", {}).get("view", 0),
                "like_count": v.get("stat", {}).get("like", 0),
                "coin_count": v.get("stat", {}).get("coin", 0),
                "favorite_count": v.get("stat", {}).get("favorite", 0),
                "duration": v.get("duration", 0),
                "bilibili_category": v.get("tname", ""),
                "classification": result["classification"] if result else None,
                "ai_tags": result["tags"] if result else [],
            }
            classified_videos.append(video_data)
        
        # 趋势分析
        classifier = get_classifier()
        trend = await loop.run_in_executor(None, classifier.analyze_trend, classified_videos)
        
        # 生成洞察
        insights = _generate_insights(trend, classified_videos)
        
        # 构建完整结果
        result = {
            "analysis_date": datetime.now().isoformat(),
            "sample_size": len(classified_videos),
            "trend_analysis": trend,
            "key_insights": insights,
            "top_videos_by_category": _get_top_videos_by_category(classified_videos),
            "videos": classified_videos[:20],  # 包含前20个视频详情
        }
        
        # 生成可视化报告
        report_files = {}
        if generate_report and REPORT_AVAILABLE:
            try:
                generator = ReportGenerator(output_dir="reports")
                report_files = await loop.run_in_executor(
                    None, 
                    generator.generate_all_reports,
                    result,
                    classified_videos
                )
                result["report_files"] = report_files
            except Exception as e:
                result["report_error"] = str(e)
        
        return result
        
    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}