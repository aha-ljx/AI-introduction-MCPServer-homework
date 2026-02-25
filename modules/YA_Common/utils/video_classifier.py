"""
视频内容分类器 - XGBoost集成学习版本
"""
import jieba
import jieba.analyse
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os


class VideoClassifier:
    """
    基于集成学习的视频内容分类器
    使用 VotingClassifier (XGBoost + RandomForest + LogisticRegression)
    """
    
    # 保留原有分类体系
    CATEGORIES = [
        "游戏", "科技", "美食", "生活", "娱乐", 
        "知识", "音乐", "动画", "影视", "运动", 
        "时尚", "汽车", "其他"
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化分类器
        
        Args:
            model_path: 预训练模型路径，为None则使用规则回退
        """
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.ensemble_model: Optional[VotingClassifier] = None
        self.is_trained = False
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # 未训练状态，使用规则回退
            self._init_rule_fallback()
    
    def _init_rule_fallback(self):
        """初始化规则回退（兼容旧版本）"""
        self.fallback_keywords = {
            "游戏": ["游戏", "电竞", "Steam", "原神", "王者", "LOL", "黑神话"],
            "科技": ["科技", "数码", "AI", "人工智能", "编程", "代码", "手机"],
            "美食": ["美食", "吃播", "料理", "烹饪", "探店", "火锅"],
            "生活": ["生活", "Vlog", "日常", "RoomTour", "宠物"],
            "娱乐": ["娱乐", "搞笑", "综艺", "明星", "鬼畜", "吐槽"],
            "知识": ["知识", "学习", "历史", "科普", "教程", "纪录片"],
            "音乐": ["音乐", "歌曲", "演唱", "翻唱", "MV", "乐器"],
            "动画": ["动画", "动漫", "二次元", "番剧", "MAD", "Vtuber"],
            "影视": ["电影", "电视剧", "影评", "解说", "Netflix", "漫威"],
            "运动": ["运动", "健身", "篮球", "足球", "NBA", "瑜伽"],
            "时尚": ["时尚", "穿搭", "美妆", "护肤", "化妆", "OOTD"],
            "汽车": ["汽车", "电动车", "特斯拉", "比亚迪", "赛车"],
        }
        for keywords in self.fallback_keywords.values():
            for word in keywords:
                jieba.add_word(word)
    
    def _create_ensemble(self) -> VotingClassifier:
        """
        创建集成模型：XGBoost + RandomForest + LogisticRegression
        使用软投票（概率加权）
        """
        # XGBoost（梯度提升树）
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # RandomForest（Bagging）
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # LogisticRegression（线性基线）
        lr = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        # 软投票集成（根据概率加权）
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb),
                ('rf', rf),
                ('lr', lr)
            ],
            voting='soft',  # 软投票比硬投票更准确
            weights=[2, 1, 1]  # XGBoost权重更高
        )
        
        return ensemble
    
    def preprocess_text(self, title: str, description: str = "", tname: str = "") -> str:
        """
        文本预处理：分词 + 清洗
        """
        text = f"{title} {description} {tname}"
        # 精确模式分词
        words = jieba.lcut(text, cut_all=False)
        # 过滤停用词和单字
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = [w for w in words if len(w) > 1 and w not in stopwords]
        return " ".join(words)
    
    def train(self, texts: List[str], labels: List[str], save_path: Optional[str] = None):
        """
        训练集成模型
        
        Args:
            texts: 文本列表（已预处理的标题+描述）
            labels: 对应分类标签
            save_path: 模型保存路径
        """
        from sklearn.model_selection import train_test_split
        from collections import Counter
        # 统计各类别数量
        label_counts = Counter(labels)
        print(f"原始类别分布: {dict(label_counts)}")
        # 过滤样本数 < 3 的类别
        min_samples = 3
        valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
        # 过滤数据
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels):
            if label in valid_labels:
                filtered_texts.append(text)
                filtered_labels.append(label)
    
        removed_labels = set(labels) - valid_labels
        print(f"过滤后: {len(filtered_texts)} 条样本，{len(valid_labels)} 个类别")
        if removed_labels:
            print(f"移除的类别（样本不足{min_samples}个）: {removed_labels}")
    
        # 检查是否还有足够数据
        if len(valid_labels) < 2:
            raise ValueError(f"有效类别不足2个（当前{len(valid_labels)}个），无法训练。请增加样本数量。")
    
        if len(filtered_texts) < 10:
            raise ValueError(f"有效样本不足10个（当前{len(filtered_texts)}个），无法训练。")
    
    # 划分训练集和验证集
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                filtered_texts, filtered_labels, 
                test_size=0.2, 
                random_state=42, 
                stratify=filtered_labels
            )
        except ValueError as e:
            # 如果 stratify 失败，改用随机划分
            print(f"警告: 分层抽样失败（{e}），改用随机划分")
            X_train, X_val, y_train, y_val = train_test_split(
                filtered_texts, filtered_labels, 
                test_size=0.2, 
                random_state=42
            )
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # 最大特征数
            ngram_range=(1, 2),      # 使用unigram和bigram
            min_df=2,                # 最小文档频率
            max_df=0.95,             # 最大文档频率
            sublinear_tf=True        # 使用1+log(tf)
        )
        
        # 拟合向量化器
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        # 创建并训练集成模型
        self.ensemble_model = self._create_ensemble()
        self.ensemble_model.fit(X_train_vec, y_train)
        
        # 验证准确率
        train_acc = self.ensemble_model.score(X_train_vec, y_train)
        val_acc = self.ensemble_model.score(X_val_vec, y_val)
        
        self.is_trained = True
        
        print(f"训练完成 - 训练集准确率: {train_acc:.4f}, 验证集准确率: {val_acc:.4f}")
        
        # 保存模型
        if save_path:
            self._save_model(save_path)
        
        return {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "model_type": "ensemble_voting",
            "base_learners": ["XGBoost", "RandomForest", "LogisticRegression"]
        }
    
    def predict(self, title: str, description: str = "", tname: str = "") -> Dict[str, Any]:
        """
        预测视频分类（集成模型版）
        """
        # 如果模型未训练，回退到规则版本
        if not self.is_trained:
            return self._rule_predict(title, description, tname)
        
        # 预处理文本
        processed_text = self.preprocess_text(title, description, tname)
        
        # 向量化
        X = self.vectorizer.transform([processed_text])
        
        # 获取各分类器概率
        probabilities = self.ensemble_model.predict_proba(X)[0]
        
        # 获取Top3
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {"category": self.ensemble_model.classes_[i], "score": round(float(probabilities[i]), 4)}
            for i in top3_indices
        ]
        
        primary_idx = top3_indices[0]
        
        return {
            "primary": self.ensemble_model.classes_[primary_idx],
            "confidence": round(float(probabilities[primary_idx]), 4),
            "all_scores": {
                cat: round(float(prob), 4) 
                for cat, prob in zip(self.ensemble_model.classes_, probabilities)
            },
            "top3": top3,
            "keywords": self._extract_keywords(title, description),
            "bilibili_tname": tname,
            "method": "ensemble_xgb_rf_lr",  # 标记为集成方法
            "is_fallback": False
        }
    
    def _rule_predict(self, title: str, description: str, tname: str) -> Dict[str, Any]:
        """
        规则回退版本（保持与原代码兼容）
        """
        text = f"{title} {description} {tname}"
        words = jieba.lcut(text)
        
        scores = {}
        for category, indicators in self.fallback_keywords.items():
            base_score = sum(2 for w in words if w in indicators)
            title_score = sum(5 for w in jieba.lcut(title) if w in indicators)
            scores[category] = base_score + title_score
        
        if sum(scores.values()) == 0:
            return {
                "primary": "其他",
                "confidence": 1.0,
                "all_scores": {k: 0 for k in self.fallback_keywords.keys()},
                "keywords": [],
                "method": "fallback_rule",
                "is_fallback": True
            }
        
        total = sum(scores.values())
        normalized = {k: round(v/total, 3) for k, v in scores.items()}
        primary = max(scores, key=scores.get)
        
        top3 = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "primary": primary,
            "confidence": normalized[primary],
            "all_scores": normalized,
            "top3": [{"category": k, "score": v} for k, v in top3],
            "keywords": self._extract_keywords(title, description),
            "bilibili_tname": tname,
            "method": "fallback_rule",
            "is_fallback": True
        }
    
    def _extract_keywords(self, title: str, description: str, num: int = 5) -> List[str]:
        """提取关键词"""
        text = f"{title} {description}"
        keywords = jieba.analyse.extract_tags(text, topK=num*2, withWeight=True)
        return [k for k, _ in keywords if len(k) > 1][:num]
    
    def extract_tags(self, title: str, description: str = "", num_tags: int = 5) -> List[Dict[str, Any]]:
        """保持与原接口兼容"""
        text = f"{title} {description}"
        keywords = jieba.analyse.extract_tags(text, topK=num_tags*2, withWeight=True)
        
        tags = []
        for word, weight in keywords:
            if len(word) < 2 or word.isdigit():
                continue
            if word in ["视频", "这个", "一个", "什么", "今天"]:
                continue
            tags.append({"tag": word, "weight": round(weight, 3), "source": "tfidf"})
            if len(tags) >= num_tags:
                break
        
        return tags
    
    def analyze_trend(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """趋势分析（保持兼容）"""
        if not videos:
            return {"error": "Empty video list"}
        
        category_dist = {}
        for v in videos:
            cat = v.get("classification", {}).get("primary", "未知")
            category_dist[cat] = category_dist.get(cat, 0) + 1
        
        total = len(videos)
        distribution = {
            k: {"count": v, "percentage": round(v/total*100, 1)}
            for k, v in sorted(category_dist.items(), key=lambda x: x[1], reverse=True)
        }
        
        all_tags = []
        for v in videos:
            all_tags.extend(v.get("tags", []))
        
        tag_freq = {}
        for t in all_tags:
            tag = t["tag"]
            tag_freq[tag] = tag_freq.get(tag, 0) + t["weight"]
        
        top_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 统计使用集成模型的比例
        ensemble_count = sum(1 for v in videos if not v.get("classification", {}).get("is_fallback", True))
        
        return {
            "total_videos": total,
            "category_distribution": distribution,
            "dominant_category": list(distribution.keys())[0] if distribution else None,
            "top_tags": [{"tag": k, "score": round(v, 3)} for k, v in top_tags],
            "diversity": len(category_dist) / len(self.CATEGORIES),
            "model_coverage": round(ensemble_count / total * 100, 1) if total > 0 else 0
        }
    
    def _save_model(self, path: str):
        """保存模型"""
        model_data = {
            'vectorizer': self.vectorizer,
            'ensemble_model': self.ensemble_model,
            'categories': self.CATEGORIES
        }
        joblib.dump(model_data, path)
        print(f"模型已保存到: {path}")
    
    def _load_model(self, path: str):
        """加载模型"""
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.ensemble_model = model_data['ensemble_model']
        self.CATEGORIES = model_data.get('categories', self.CATEGORIES)
        self.is_trained = True
        print(f"模型已从 {path} 加载")


# 全局单例
_classifier_instance: Optional[VideoClassifier] = None


def get_classifier(model_path: Optional[str] = None) -> VideoClassifier:
    """
    获取分类器单例
    
    Args:
        model_path: 可选，指定模型路径。为None则自动查找项目根目录的ensemble_model.pkl
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        # 如果未指定路径，自动查找项目根目录
        if model_path is None:
            # 获取当前文件路径: modules/YA_Common/utils/video_classifier.py
            current_file = os.path.abspath(__file__)
            
            # 向上回溯到项目根目录 (YA_MCPSERVER_TEMPLATE/)
            # 当前: modules/YA_Common/utils/video_classifier.py
            # 目标: YA_MCPSERVER_TEMPLATE/ensemble_model.pkl
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(current_file)  # 上溯4层
                    )
                )
            )
            
            model_path = os.path.join(project_root, "ensemble_model.pkl")
            
            # 如果模型文件不存在，使用None（触发训练模式/规则回退）
            if not os.path.exists(model_path):
                print(f"[Warning] 模型文件不存在: {model_path}，将使用规则回退模式")
                model_path = None
        
        _classifier_instance = VideoClassifier(model_path=model_path)
    
    return _classifier_instance