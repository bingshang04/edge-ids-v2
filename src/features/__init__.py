"""
Edge-IDS 特征提取模块（48维）
"""
from .flow_features import FeatureExtractor, FlowStats, FEATURE_DIM, create_feature_extractor

__all__ = ['FeatureExtractor', 'FlowStats', 'FEATURE_DIM', 'create_feature_extractor']
