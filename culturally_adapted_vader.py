import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer #, NEGATE, BOOSTER_DICT
import copy
import pandas as pd

# Make sure VADER is installed
nltk.download('vader_lexicon')

class ChineseAdaptedVader:
    def __init__(self):
        # 初始化原始 VADER 分析器    original VADER analyzer
        self.original_analyzer = SentimentIntensityAnalyzer()

        # 创建分析器副本，以供调整
        self.analyzer = SentimentIntensityAnalyzer()  # 创建 VADER 情感分析器的实例 `self.analyzer`
        self.lexicon = self.analyzer.lexicon.copy()   # 复制 lexicon 副本，以供修改
        
        # 添加文化适应
        self.add_cultural_terms()                # 文化
        self.modify_emotional_display_norms()    # 情感表达规范
        self.add_collectivist_terms()            # 集体主义
        self.add_respect_terms()                 # 尊重（长辈/权威）
        self.modify_idioms_and_expressions()     # 成语/特殊表达
        
        # 更新分析器 lexicon
        self.analyzer.lexicon = self.lexicon
        
        # 文化背景调节参数 - use these when analyzing text
        self.modesty_scale = 1.2   # 自夸负面评分（谦虚）
        self.indirect_scale = 0.8  # 减弱间接表达强度（礼貌）
        
    def add_cultural_terms(self):
        """添加文化特有词汇 that may be missing from VADER"""
        Chinese_cultural_terms = {
            # 家庭/社会关系
            'family': 2.5,       # 家庭（高度正面）
            'filial': 2.0,       # 孝道（高度正面）
            'ancestor': 1.5,     # 祖先
            'elder': 0.6,        # 长辈
            'harmony': 2.0,      # 社会和谐（重要）
            'face': 0.0,         # Neutral but contextually important
            'shame': -2.5,       # 羞耻（比西方负面）
            'honor': 2.5,        # 光荣（比西方正面）
            'duty': 1.0,         # 责任（正面）
            'peace': 2.0,        # 和平
            'lineage': 1.8,      # 传承（正面）
            
            # 表达方式
            'possibly': -0.3,
            'somewhat': -0.4,    # 委婉表达（轻微负面）
            'perhaps': -0.5,     # 间接表达（可能负面）
            'consider': -0.2,    # （重要）
            'seem': -0.2,        # 间接表达（可能表示保留）
            'hopefully': 0.3,    # 含蓄希望（中性偏正面）
            'maybe': -0.6,
            
            # 文化概念
            'fate': 0.0,         # （重要）
            'luck': 2.0,         # （重要）
            'fortune': 1.8,
            'destiny': 0.5,
            'karma': 0.0,        # 因果（中立，重要）
            'auspicious': 2.0,
            'prosperous': 2.0,
            
            # 价值观
            'modest': 1.0,       # 谦虚（高度正面）
            'humble': 0.8,       # 谦虚
            'perseverance': 2.0, # 毅力
            'endurance': 1.5,    # 耐力
            'balance': 1.5,      # 平衡
            'apologize': 0.5     # 道歉
        }
        
        # Add these terms to the lexicon
        self.lexicon.update(Chinese_cultural_terms)
        
    def modify_emotional_display_norms(self):
        """调整分数（情绪展示术语）"""
        # In Chinese cultures, emotional restraint may be valued
        # While extreme displays might be viewed differently than in Western contexts
        
        emotion_display_terms = {
            # 约束 - 减弱积极情绪
            'reserved': 0.5,     # 保留 Less negative
            'composed': 1.2,     # 组成 More positive
            
            # 情感展示
            'excited': 0.3,      # Less positive
            'enthusiastic': -1.2, # Less positive
            'loud': -1.0,        # More negative
            'boisterous': -1.2   # More negative
        }
        
        # 更新 lexicon
        for term, score in emotion_display_terms.items():
            self.lexicon[term] = score
            
    def add_collectivist_terms(self):
        """集体主义价值观"""
        collectivist_terms = {
            'we': 0.5,
            'our': 0.5,
            'together': 1.5,
            'community': 1.8,
            'collective': 1.8,
            'group': 0.8,
            'teamwork': 1.8,
            'cooperation': 1.8,
            'consensus': 1.5,
            'unity': 2.0,

            # 减弱个人主义
            'independent': 0.5,  # Less positive than in Western contexts
            'individual': -0.3,   # Neutral
            'self': -0.3,         # Neutral
            'unique': 0.0,       # Less positive
            'standout': 0.0,     # Neutral (can be negative in some contexts)
            'selfish': -2.0
        }
        
        # Update these terms in the lexicon
        self.lexicon.update(collectivist_terms)
        
    def add_respect_terms(self):
        """尊重（长辈/权威/制度）"""
        respect_terms = {
            'respect': 2.0,
            'honor': 2.0,
            'revere': 1.8,
            'elder': 1.0,
            'senior': 0.8,
            'teacher': 1.0,
            'master': 1.0,
            'authority': 0.8,
            'tradition': 1.0,
            'ritual': 1.2,
            'ceremony': 1.0,
            'festival': 2.0,
            'celebrate': 1.8,
            
            # 挑战权威
            'challenge': -0.5,   # More negative in hierarchical contexts
            'confront': -1.5,    # More negative
            'disagree': -1.0    # More negative
        }
        
        # Update these terms in the lexicon
        self.lexicon.update(respect_terms)
        
    def modify_idioms_and_expressions(self):
        """Add or modify Chinese idioms and expressions"""
        # These would ideally be extensive and language-specific
        idioms = {
            # Examples (would be more extensive in practice)
            'bamboo': 1.5,       # 竹
            'dragon': 2.0,       # 龙（东方正面，西方负面）
            'phoenix': 1.8,
            'tiger': 0.8,
            'jade': 1.5,
            'mountain': 0.8,
            'water': 0.8,
            'moon': 0.8,
            'wind': 0.5,
            'tea': 0.8
        }
        
        # Update these terms in the lexicon
        self.lexicon.update(idioms)
    
    def analyze(self, text, cultural_context='general'):
        """
        Analyze the sentiment of text with cultural context adjustments
        
        Parameters:
        -----------
        text : str
            The text to analyze
        cultural_context : str
            The specific cultural context ('chinese')
            Default is 'general' for pan-Asian adjustments
            
        Returns:
        --------
        dict
            A dictionary with sentiment scores
        """
        # Get the base VADER scores
        scores = self.analyzer.polarity_scores(text)
        
        # Apply cultural context modifiers
        # Chinese-specific modifiers
        if self._contains_indirect_expressions(text):
            scores['neg'] *= self.indirect_scale
                
        # Adjust for modesty norms if text contains self-references
        if self._contains_self_references(text):
            scores['pos'] /= self.modesty_scale

        
        # Recalculate compound score after adjustments
        # This is a simplified version - VADER uses a more complex formula
        compound = scores['pos'] - scores['neg']
        scores['compound'] = max(min(compound, 1.0), -1.0)
        
        # Compare with original VADER for reference
        original_scores = self.original_analyzer.polarity_scores(text)
        scores['original_compound'] = original_scores['compound']
        
        return scores
    
    def _contains_indirect_expressions(self, text):
        """Check if text contains indirect expressions"""
        indirect_markers = ['perhaps', 'maybe', 'seem', 'appear', 'might', 
                           'possibly', 'somewhat', 'rather', 'quite']
        return any(marker in text.lower() for marker in indirect_markers)
    
    def _contains_self_references(self, text):
        """Check if text contains self-references"""
        self_refs = ['i ', 'me', 'my', 'mine', 'myself']
        return any(ref in text.lower() for ref in self_refs)
    
    def _adjust_for_politeness(self, text, scores):
        """Adjust scores based on politeness indicators"""
        politeness_markers = ['please', 'thank', 'sorry', 'excuse']
        politeness_count = sum(text.lower().count(marker) for marker in politeness_markers)
        
        if politeness_count > 0:
            scores['pos'] += (0.1 * politeness_count)
        
        return scores
    
    def _adjust_for_honorifics(self, text, scores):
        """Adjust scores based on presence of honorific language"""
        honorific_markers = ['sir', 'madam', 'mr', 'mrs', 'miss', 'dr', 'professor']
        honorific_count = sum(text.lower().count(marker) for marker in honorific_markers)
        
        if honorific_count > 0:
            scores['pos'] += (0.05 * honorific_count)
        
        return scores
        
    def compare_with_original(self, texts):
        """Compare adjusted sentiment with original VADER"""
        results = []
        
        for text in texts:
            adapted = self.analyze(text)
            original = self.original_analyzer.polarity_scores(text)
            
            results.append({
                'text': text,
                'original_compound': original['compound'],
                'adapted_compound': adapted['compound'],
                'difference': adapted['compound'] - original['compound']
            })
            
        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Create the culturally-adapted analyzer
    Chinese_vader = ChineseAdaptedVader()

    # # Replace with your file path and column name
    # file_path = "lyrics_dataset.xlsx"
    # df = pd.read_excel(file_path)

    example_texts = [
        "I am very proud of my achievements and want everyone to know.",
        "Perhaps we should consider this option, if it's not too much trouble.",
        "The team worked together harmoniously to achieve our goals.",
        "He directly challenged his manager's decision in front of everyone.",
        "I humbly accept this award on behalf of my family and teachers.",
        "The dragon brings good fortune and prosperity to all.",
        "I'm sorry to disturb you, but I might need some assistance.",
        "He maintained perfect silence throughout the ceremony.",
        "My individual creativity makes me stand out from the group."
    ]


    
    # Analyze and print results
    print("Comparing Original VADER vs. Chinese-Adapted VADER:\n")
    
    for lyrics in example_texts:
        original = Chinese_vader.original_analyzer.polarity_scores(lyrics)
        adapted = Chinese_vader.analyze(lyrics)
        
        print(f"Text: \"{lyrics}\"")
        print(f"  Original VADER: {original['compound']:.3f}")
        print(f"  Asian-Adapted:  {adapted['compound']:.3f}")
        print(f"  Difference:     {adapted['compound'] - original['compound']:.3f}")
        print()
    
    # Compare in bulk and get a DataFrame for analysis
    comparison_df = Chinese_vader.compare_with_original(example_texts)
    print("\nAggregate Comparison:")
    print(comparison_df.describe())
