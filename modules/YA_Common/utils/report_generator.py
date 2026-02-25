"""
可视化报告生成器
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template


class ReportGenerator:
    """B站热门视频分析报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, data: Dict[str, Any], videos: List[Dict]) -> Dict[str, str]:
        """
        生成所有格式的报告
        
        Returns:
            生成的文件路径字典
        """
        files = {}
        
        # 1. JSON数据
        files['json'] = self._save_json(data)
        
        # 2. 图表
        files['charts'] = self._generate_charts(data, videos)
        
        # 3. HTML报告
        files['html'] = self._generate_html(data, videos)
        
        # 4. Markdown文本报告
        files['markdown'] = self._generate_markdown(data, videos)
        
        print(f"\n报告生成完成:")
        for k, v in files.items():
            print(f"  {k}: {v}")
        
        return files
    
    def _save_json(self, data: Dict) -> str:
        """保存原始JSON数据"""
        filepath = self.output_dir / f"bilibili_report_{self.timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(filepath)
    
    def _generate_charts(self, data: Dict, videos: List[Dict]) -> List[str]:
        """生成可视化图表"""
        chart_files = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 分类分布饼图
        if 'trend_analysis' in data and 'category_distribution' in data['trend_analysis']:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            categories = data['trend_analysis']['category_distribution']
            labels = list(categories.keys())
            sizes = [v['count'] for v in categories.values()]
            colors = plt.cm.Set3(range(len(labels)))
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90
            )
            ax.set_title('B站热门视频分类分布', fontsize=16, pad=20)
            
            filepath = self.output_dir / f"category_pie_{self.timestamp}.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(str(filepath))
        
        # 2. 播放量TOP10柱状图
        if videos:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            top_videos = sorted(videos, key=lambda x: x.get('play_count', 0), reverse=True)[:10]
            titles = [v['title'][:15] + '...' if len(v['title']) > 15 else v['title'] 
                    for v in top_videos]
            plays = [v.get('play_count', 0) / 10000 for v in top_videos]  # 转为万
            
            bars = ax.barh(range(len(titles)), plays, color='skyblue')
            ax.set_yticks(range(len(titles)))
            ax.set_yticklabels(titles)
            ax.set_xlabel('播放量（万）', fontsize=12)
            ax.set_title('热门视频播放量TOP10', fontsize=16)
            ax.invert_yaxis()
            
            # 添加数值标签
            for i, (bar, play) in enumerate(zip(bars, plays)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{play:.1f}万', va='center', fontsize=9)
            
            filepath = self.output_dir / f"top10_plays_{self.timestamp}.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(str(filepath))
        
        # 3. 互动数据对比图（点赞/投币/收藏）
        if videos:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sample = videos[:10]  # 取前10个
            x = range(len(sample))
            width = 0.25
            
            likes = [v.get('like_count', 0) / 10000 for v in sample]
            coins = [v.get('coin_count', 0) / 10000 for v in sample]
            favorites = [v.get('favorite_count', 0) / 10000 for v in sample]
            
            ax.bar([i - width for i in x], likes, width, label='点赞', color='#FF6B6B')
            ax.bar(x, coins, width, label='投币', color='#4ECDC4')
            ax.bar([i + width for i in x], favorites, width, label='收藏', color='#45B7D1')
            
            ax.set_ylabel('数量（万）', fontsize=12)
            ax.set_title('视频互动数据对比（前10个）', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels([v['title'][:8] + '...' for v in sample], rotation=45, ha='right')
            ax.legend()
            
            filepath = self.output_dir / f"interaction_compare_{self.timestamp}.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            chart_files.append(str(filepath))
        
        return chart_files
    
    def _generate_html(self, data: Dict, videos: List[Dict]) -> str:
        """生成美观的HTML报告"""
        
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>B站热门视频分析报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .content { padding: 40px; }
        .section {
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label { color: #666; margin-top: 5px; }
        .category-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .category-tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .video-grid {
            display: grid;
            gap: 20px;
        }
        .video-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            align-items: start;
        }
        .video-thumb {
            width: 120px;
            height: 80px;
            background: #ddd;
            border-radius: 8px;
            flex-shrink: 0;
            object-fit: cover;
        }
        .video-info { flex: 1; }
        .video-title {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .video-meta {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 8px;
        }
        .video-stats {
            display: flex;
            gap: 15px;
            font-size: 0.85em;
            color: #888;
        }
        .ai-tag {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .insights {
            background: #fff3cd;
            border-left-color: #ffc107;
        }
        .insight-item {
            padding: 10px 0;
            border-bottom: 1px dashed #ddd;
        }
        .insight-item:last-child { border-bottom: none; }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 B站热门视频分析报告</h1>
            <p>生成时间：{{ timestamp }} | 样本数量：{{ total_videos }}个视频</p>
        </div>
        
        <div class="content">
            <!-- 关键指标 -->
            <div class="section">
                <h2>📈 关键指标</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{{ total_videos }}</div>
                        <div class="stat-label">分析视频数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ total_plays // 10000 }}万</div>
                        <div class="stat-label">总播放量</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ category_count }}</div>
                        <div class="stat-label">内容分类数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{{ avg_duration // 60 }}分</div>
                        <div class="stat-label">平均时长</div>
                    </div>
                </div>
            </div>

            <!-- 分类分布 -->
            {% if categories %}
            <div class="section">
                <h2>🎯 内容分类分布</h2>
                <div class="category-list">
                    {% for cat, info in categories.items() %}
                    <span class="category-tag">{{ cat }} ({{ info.percentage }}%)</span>
                    {% endfor %}
                </div>
                {% if chart_files %}
                <div class="chart-container">
                    <img src="{{ chart_files[0] }}" alt="分类分布图">
                </div>
                {% endif %}
            </div>
            {% endif %}

            <!-- 趋势洞察 -->
            {% if insights %}
            <div class="section insights">
                <h2>💡 趋势洞察</h2>
                {% for insight in insights %}
                <div class="insight-item">• {{ insight }}</div>
                {% endfor %}
            </div>
            {% endif %}

            <!-- 热门视频 -->
            <div class="section">
                <h2>🔥 热门视频TOP10</h2>
                <div class="video-grid">
                    {% for video in top_videos %}
                    <div class="video-card">
                        <img class="video-thumb" src="{{ video.pic }}" alt="封面" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22120%22 height=%2280%22><rect fill=%22%23ddd%22 width=%22120%22 height=%2280%22/></svg>'">
                        <div class="video-info">
                            <div class="video-title">{{ video.title }}</div>
                            <div class="video-meta">
                                👤 {{ video.author }} | 📁 {{ video.bilibili_category }}
                                {% if video.ai_classification %}
                                | 🤖 AI分类: {{ video.ai_classification.primary }}
                                {% endif %}
                            </div>
                            <div class="video-stats">
                                ▶️ {{ video.play_count // 10000 }}万播放 
                                👍 {{ video.like_count // 10000 }}万赞 
                                💰 {{ video.coin_count // 10000 }}万币
                            </div>
                            {% if video.ai_tags %}
                            <div style="margin-top: 8px;">
                                {% for tag in video.ai_tags[:3] %}
                                <span class="ai-tag">{{ tag.tag }}</span>
                                {% endfor %}
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <!-- 图表展示 -->
            {% if chart_files|length > 1 %}
            <div class="section">
                <h2>📊 数据可视化</h2>
                {% for chart in chart_files[1:] %}
                <div class="chart-container">
                    <img src="{{ chart }}" alt="数据图表">
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>

        <div class="footer">
            <p>由 Hello-YA-MCP-Server 自动生成 | 数据来源：哔哩哔哩</p>
        </div>
    </div>
</body>
</html>
        """
        
        # 准备数据
        trend = data.get('trend_analysis', {})
        categories = trend.get('category_distribution', {})
        
        # 计算统计数据
        total_plays = sum(v.get('play_count', 0) for v in videos)
        avg_duration = sum(v.get('duration', 0) for v in videos) // len(videos) if videos else 0
        
        # 获取图表文件（相对路径）
        chart_files = [f"category_pie_{self.timestamp}.png",
                      f"top10_plays_{self.timestamp}.png",
                      f"interaction_compare_{self.timestamp}.png"]
        chart_files = [str(self.output_dir / f) for f in chart_files 
                      if (self.output_dir / f).exists()]
        
        template = Template(template_str)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_videos=len(videos),
            total_plays=total_plays,
            category_count=len(categories),
            avg_duration=avg_duration,
            categories=categories,
            insights=trend.get('key_insights', []),
            top_videos=videos[:10],
            chart_files=chart_files
        )
        
        filepath = self.output_dir / f"report_{self.timestamp}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _generate_markdown(self, data: Dict, videos: List[Dict]) -> str:
        """生成Markdown文本报告"""
        
        trend = data.get('trend_analysis', {})
        categories = trend.get('category_distribution', {})
        
        lines = [
            "# 📊 B站热门视频分析报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**样本数量**: {len(videos)} 个视频",
            "",
            "## 📈 关键指标",
            "",
            f"- 分析视频数: {len(videos)}",
            f"- 总播放量: {sum(v.get('play_count', 0) for v in videos) // 10000} 万",
            f"- 内容分类数: {len(categories)}",
            f"- 平均时长: {sum(v.get('duration', 0) for v in videos) // len(videos) // 60 if videos else 0} 分钟",
            "",
            "## 🎯 分类分布",
            "",
            "| 分类 | 数量 | 占比 |",
            "|------|------|------|",
        ]
        
        for cat, info in sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True):
            lines.append(f"| {cat} | {info['count']} | {info['percentage']}% |")
        
        lines.extend([
            "",
            "## 💡 趋势洞察",
            "",
        ])
        
        for insight in trend.get('key_insights', []):
            lines.append(f"- {insight}")
        
        lines.extend([
            "",
            "## 🔥 热门视频TOP10",
            "",
            "| 排名 | 标题 | 作者 | 播放量 | 分类 |",
            "|------|------|------|--------|------|",
        ])
        
        for i, v in enumerate(sorted(videos, key=lambda x: x.get('play_count', 0), reverse=True)[:10], 1):
            title = v['title'][:30] + '...' if len(v['title']) > 30 else v['title']
            ai_cat = v.get('ai_classification', {}).get('primary', '未分类')
            lines.append(f"| {i} | {title} | {v.get('author', '未知')} | {v.get('play_count', 0) // 10000}万 | {ai_cat} |")
        
        lines.extend([
            "",
            "## 🏷️ 热门标签",
            "",
        ])
        
        top_tags = trend.get('top_tags', [])[:10]
        for tag in top_tags:
            lines.append(f"- **{tag['tag']}** (热度: {tag['score']:.2f})")
        
        lines.extend([
            "",
            "---",
            "*由 Hello-YA-MCP-Server 自动生成*",
        ])
        
        content = '\n'.join(lines)
        
        filepath = self.output_dir / f"report_{self.timestamp}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)