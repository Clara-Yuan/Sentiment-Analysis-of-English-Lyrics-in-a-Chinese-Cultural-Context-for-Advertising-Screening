import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Define emotion categories
EMOTION_CATEGORIES = ['excited', 'negative', 'romantic', 'peaceful', 'positive']


def preprocess_text(text):
    """Clean and preprocess text data"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    return ""


def load_data(file_path):
    """Load and preprocess data"""
    df = pd.read_excel(file_path)
    df['processed_lyrics'] = df['lyrics'].apply(preprocess_text)
    return df


def train_models(df):
    """Train non-sentiment classifiers without saving"""
    models = {}

    for emotion in EMOTION_CATEGORIES:
        print(f"\n=== Training non-{emotion} classifier ===")

        # Create binary labels
        # 添加一个新的二元列`non_{emotion}`，如果情感属于sentiment，则标记为 `0`，否则non-sentiment标记为 `1`。
        df[f'non_{emotion}'] = df['label'].apply(lambda x: 0 if x == emotion else 1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_lyrics'],
            df[f'non_{emotion}'],
            test_size=0.2,
            stratify=df[f'non_{emotion}'],
            random_state=42
        )

        # Build and train pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(     # 将文本数据转换为 TF-IDF 特征
                max_features=5000,
                ngram_range=(1, 2))),
             ('clf', LogisticRegression(   # 逻辑回归
                 class_weight='balanced',
                 solver='liblinear'))
             ])

        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred,
                                    target_names=[f"{emotion}", f"non_{emotion}"]))

        # Store model in memory
        models[emotion] = pipeline

    return models


def analyze_data(models, df):
    """Generate predictions and visualizations"""
    result_df = df.copy()

    # Generate predictions
    for emotion, model in models.items():
        result_df[f'non_{emotion}_prob'] = model.predict_proba(df['processed_lyrics'])[:, 1]

        # 仅对未标注数据生成预测标签（已标注数据保留原始标签）
        mask_unlabeled = df['label'].isna()  # 假设未标注数据的label为空
        result_df.loc[mask_unlabeled, f'non_{emotion}_pred'] = model.predict(df.loc[mask_unlabeled, 'processed_lyrics'])
        # 已标注数据保持原始标签（假设原始标签列名为 'label'）
        result_df.loc[~mask_unlabeled, f'non_{emotion}_pred'] = df.loc[~mask_unlabeled, 'label'].apply(
            lambda x: 0 if x == emotion else 1  # 与训练时的标签逻辑一致
        )

    # Visualize distributions
    plt.figure(figsize=(15, 10))
    plt.suptitle("Non-Sentiment Probability Distributions", fontsize=16)

    for idx, emotion in enumerate(EMOTION_CATEGORIES, 1):
        plt.subplot(2, 3, idx)
        sns.histplot(
            result_df[f'non_{emotion}_prob'],
            bins=20,
            kde=True,
            stat='probability',
            color='royalblue'
        )
        plt.axvline(0.5, color='red', linestyle='--', linewidth=1)
        plt.title(f'Non-{emotion.capitalize()}')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Proportion')

    plt.tight_layout()
    plt.show()

    return result_df


""" 绘制非情感概率 雷达图 """
def plot_radar_chart(df, save_path=None):
    """
    :param df: 必须包含 non_{emotion}_prob 列（emotion来自EMOTION_CATEGORIES）
    :param save_path: 图片保存路径（可选）
    """
    # ========== 与趋势图相同的列检查 ==========
    required_cols = [f'non_{e}_prob' for e in EMOTION_CATEGORIES]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}, 请先运行 analyze_data()")

    plt.figure(figsize=(10, 10))

    # ========== 数据计算部分 ==========
    # 计算平均概率
    avg_probs = [df[f'non_{e}_prob'].mean() for e in EMOTION_CATEGORIES]
    categories = [f'Non-{e.capitalize()}' for e in EMOTION_CATEGORIES]

    # 闭合图形参数（与趋势图类似的环形处理）
    angles = np.linspace(0, 2 * np.pi, len(EMOTION_CATEGORIES), endpoint=False).tolist()
    angles += angles[:1]
    avg_probs += avg_probs[:1]
    categories += categories[:1]

    # ========== 绘图 ==========
    # 创建极坐标轴
    ax = plt.subplot(111, polar=True)
    # 绘制主数据线
    ax.plot(angles, avg_probs, 'o-', linewidth=4, color='#2c7bb6', markersize=12)
    ax.fill(angles, avg_probs, alpha=0.25, color='#2c7bb6')

    # ========== 样式设置（与趋势图保持一致的字体和网格）==========
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories[:-1], size=12, color='navy')
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # highlight: 覆盖绘制绿色80%网格线
    ax.plot(np.linspace(0, 2 * np.pi, 100),  # X轴：完整的360度角度
            [0.8] * 100,  # Y轴：固定80%值
            color='limegreen',  # 荧光绿
            linestyle='--',
            alpha=0.3,
            zorder=3)  # 确保显示在顶层

    # ========== 标题和图例（与趋势图相同的布局逻辑）==========
    overall_avg = np.mean(avg_probs[:-1])
    plt.title(f"Non-Sentiment Probability Radar (Avg: {overall_avg:.1%})",
              y=1.1, fontsize=14, fontweight='bold')

    # 与趋势图相同的保存逻辑
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



""" 非情感概率随年份变化趋势 """

from matplotlib.colors import LinearSegmentedColormap

def plot_non_sentiment_prob_trends(df, save_path=None, window=5):
    """绘制非情感概率趋势图，仅将non-negative标绿，其他保持不变"""
    # 确保年份为整数并限定范围
    df['year'] = df['year'].astype(int)
    df = df[(df['year'] >= 2005) & (df['year'] <= 2024)]

    # 计算每年各类别的平均概率
    yearly_avg = df.groupby('year')[[f'non_{emotion}_prob' for emotion in EMOTION_CATEGORIES]].mean().reset_index()

    # 应用移动平均平滑（保持原样）
    for emotion in EMOTION_CATEGORIES:
        yearly_avg[f'non_{emotion}_prob_smooth'] = yearly_avg[f'non_{emotion}_prob'].rolling(window=window, min_periods=1).mean()

    # 计算每个类别的变化范围
    ranges = yearly_avg[[f'non_{emotion}_prob_smooth' for emotion in EMOTION_CATEGORIES]].max() - \
             yearly_avg[[f'non_{emotion}_prob_smooth' for emotion in EMOTION_CATEGORIES]].min()

    # 按变化范围降序排序情感类别
    sorted_emotions = sorted(
        EMOTION_CATEGORIES,
        key=lambda x: ranges[f'non_{x}_prob_smooth'],
        reverse=True
    )

    # 归一化变化范围到 [0, 1]（保持原样）
    min_range = ranges.min()
    max_range = ranges.max()
    normalized_ranges = (ranges - min_range) / (max_range - min_range)

    # 颜色方案：仅修改negative为绿色，其他保持原渐变色
    CUSTOM_COLORS = {
        'negative': '#b7e1a1'  # 绿色
    }
    # 其他类别仍用原渐变色
    OTHER_COLORS = ['#f0f0f0', '#e4cbff', '#a2cffe', '#004577']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', OTHER_COLORS)

    # 设置图形（完全保持原样）
    plt.figure(figsize=(12, 8))

    # 按排序后的顺序绘制折线图
    for emotion in sorted_emotions:
        col_name = f'non_{emotion}_prob_smooth'
        data = yearly_avg[['year', col_name]].dropna()

        # 仅修改颜色逻辑：negative固定绿色，其他按原规则
        if emotion == 'negative':
            line_color = CUSTOM_COLORS['negative']
        else:
            color_value = normalized_ranges[f'non_{emotion}_prob_smooth']
            line_color = cmap(color_value)

        # 完全保持原有的绘图参数（linewidth=2, alpha=0.7等）
        sns.lineplot(data=data, x='year', y=col_name,
                     color=line_color, linewidth=2, alpha=0.7,
                     label=f'Non-{emotion.capitalize()} ({ranges[f"non_{emotion}_prob_smooth"]:.2f})')

    # 完全保持原有的图形样式
    plt.title('Trends of Non-Sentiment (2005-2024)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.xticks(range(2005, 2025, 1))
    plt.ylabel('Average Non-Sentiment Probability (Smoothed)', fontsize=14)
    plt.legend(title='Non-Sentiment (Change Range)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




def main(file_path):
    """Main analysis workflow"""
    df = load_data(file_path)

    if 'label' in df.columns:
        models = train_models(df)
        results = analyze_data(models, df)
        results.to_excel(file_path.replace('.xlsx', '_results1.xlsx'), index=False)
        # prob 雷达图
        plot_radar_chart(results, save_path="Emotion Avoidance Confidence.png")
        # prob 时间序列分析函数
        plot_non_sentiment_prob_trends(results, save_path="non_sentiment_prob_trends1.png")

    else:
        print("No labels found for training")



if __name__ == "__main__":
    main(r"C:\Users\q6483\Desktop\Project\Lyrics(label).xlsx")