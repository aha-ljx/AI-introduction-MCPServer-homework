import httpx
from resources import YA_MCPServer_Resource
from typing import Any, Dict, List
import asyncio


def _sync_fetch_hot() -> Dict[str, Any]:
    """同步获取热门视频"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0",
        "Referer": "https://www.bilibili.com",
    }
    
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(
            "https://api.bilibili.com/x/web-interface/popular",
            params={"ps": 10},
            headers=headers
        )
        response.raise_for_status()
        return response.json()


def _sync_fetch_video(bvid: str) -> Dict[str, Any]:
    """同步获取视频详情"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0",
        "Referer": "https://www.bilibili.com",
    }
    
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(
            "https://api.bilibili.com/x/web-interface/view",
            params={"bvid": bvid},
            headers=headers
        )
        response.raise_for_status()
        return response.json()


@YA_MCPServer_Resource(
    "bilibili://hot/popular",
    name="bilibili_hot",
    title="Bilibili Hot Videos",
    description="获取哔哩哔哩当前热门视频列表",
    mime_type="application/json",
)
async def get_bilibili_hot() -> Any:
    """返回哔哩哔哩热门视频列表。"""
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _sync_fetch_hot)
        
        if data.get("code") != 0:
            return {"error": f"API error: {data.get('message')}"}
        
        videos = data.get("data", {}).get("list", [])
        
        formatted_videos: List[Dict[str, Any]] = []
        for video in videos:
            formatted_videos.append({
                "title": video.get("title", ""),
                "bvid": video.get("bvid", ""),
                "author": video.get("owner", {}).get("name", ""),
                "play_count": video.get("stat", {}).get("view", 0),
                "like_count": video.get("stat", {}).get("like", 0),
                "url": f"https://www.bilibili.com/video/{video.get('bvid', '')}"
            })
        
        return {
            "source": "bilibili",
            "category": "hot_videos",
            "total": len(formatted_videos),
            "videos": formatted_videos
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch: {str(e)}"}


@YA_MCPServer_Resource(
    "bilibili://video/{bvid}",
    name="bilibili_video_detail",
    title="Bilibili Video Detail",
    description="根据BV号获取哔哩哔哩视频详情",
    mime_type="application/json",
)
async def get_bilibili_video(bvid: str) -> Any:
    """返回指定BV号的哔哩哔哩视频详情。"""
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _sync_fetch_video, bvid)
        
        if data.get("code") != 0:
            return {"error": f"API error: {data.get('message')}"}
        
        video = data.get("data", {})
        
        return {
            "source": "bilibili",
            "bvid": bvid,
            "title": video.get("title", ""),
            "description": video.get("desc", ""),
            "author": video.get("owner", {}).get("name", ""),
            "play_count": video.get("stat", {}).get("view", 0),
            "like_count": video.get("stat", {}).get("like", 0),
            "url": f"https://www.bilibili.com/video/{bvid}"
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch: {str(e)}"}