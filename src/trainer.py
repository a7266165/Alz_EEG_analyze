"""
EEG 特徵提取與 XGBoost 訓練
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.analyzer import EEGFeatures, EEGAnalyzer


class FeatureExtractor:
    """
    將 EEGFeatures 轉換為 XGBoost 特徵向量
    """
    
    @staticmethod
    def extract(features: EEGFeatures) -> np.ndarray:
        """
        提取特徵向量
        
        Args:
            features: EEGFeatures 物件
            
        Returns:
            特徵向量 (238,) = 17 頻道 × 14 特徵
        """
        feature_vector = []
        
        for i in range(len(features.channels)):
            # 每個頻道的 7 個頻帶絕對能量
            for band in EEGAnalyzer.BANDS.keys():
                feature_vector.append(features.band_powers[band][i])
            
            # 每個頻道的 7 個頻帶相對能量
            for band in EEGAnalyzer.BANDS.keys():
                feature_vector.append(features.relative_powers[band][i])
        
        return np.array(feature_vector)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """
        取得特徵名稱
        
        Returns:
            特徵名稱列表
        """
        channels = EEGAnalyzer.DEFAULT_CHANNELS
        bands = list(EEGAnalyzer.BANDS.keys())
        
        names = []
        for ch in channels:
            # 絕對能量
            for band in bands:
                names.append(f"{ch}_{band}_abs")
            # 相對能量
            for band in bands:
                names.append(f"{ch}_{band}_rel")
        
        return names


class XGBoostTrainer:
    """
    XGBoost 訓練器
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        **xgb_params
    ):
        """
        Args:
            test_size: 測試集比例
            random_state: 隨機種子
            **xgb_params: XGBoost 參數
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # 預設 XGBoost 參數
        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': random_state,
            'n_jobs': -1
        }
        self.xgb_params.update(xgb_params)
    
    def prepare_data(
        self,
        features_list: List[EEGFeatures],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        準備訓練數據
        
        Args:
            features_list: EEGFeatures 列表
            labels: 標籤列表
            
        Returns:
            (X, y, subject_ids) 特徵矩陣、標籤向量和受試者ID
        """
        X = []
        subject_ids = []
        
        for features in features_list:
            feature_vector = FeatureExtractor.extract(features)
            X.append(feature_vector)
            subject_ids.append(features.subject_id)
        
        X = np.array(X)
        y = np.array(labels)
        
        print(f"特徵矩陣: {X.shape}")
        print(f"標籤向量: {y.shape}")
        print(f"受試者數: {len(set(subject_ids))} 人")
        print(f"樣本總數: {len(subject_ids)} 筆")
        print(f"類別分佈: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        return X, y, subject_ids
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        save_dir: Path = Path("workspace/analyze_result")
    ) -> Dict:
        """
        訓練 XGBoost 模型（確保同一個人的資料在同一集）
        
        Args:
            X: 特徵矩陣
            y: 標籤向量
            subject_ids: 受試者ID列表
            save_dir: 儲存目錄
            
        Returns:
            訓練結果字典
        """
        from sklearn.model_selection import GroupShuffleSplit
        
        # 使用 GroupShuffleSplit 確保同一個人的資料在同一集
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 檢查分割結果
        train_subjects = set(np.array(subject_ids)[train_idx])
        test_subjects = set(np.array(subject_ids)[test_idx])
        overlap = train_subjects & test_subjects
        
        print(f"\n訓練集: {len(X_train)} 樣本, {len(train_subjects)} 人")
        print(f"測試集: {len(X_test)} 樣本, {len(test_subjects)} 人")
        print(f"重疊受試者: {len(overlap)} 人 {'✓' if len(overlap) == 0 else '✗ 警告！'}")
        
        # 訓練模型
        print("\n訓練 XGBoost 模型...")
        model = xgb.XGBClassifier(**self.xgb_params)
        model.fit(X_train, y_train)
        
        # 預測
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        # 評估
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n訓練集準確率: {train_acc:.4f}")
        print(f"測試集準確率: {test_acc:.4f}")
        
        # 詳細報告
        print("\n=== 測試集分類報告 ===")
        print(classification_report(y_test, y_pred_test))
        
        print("\n=== 混淆矩陣 ===")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        
        # 特徵重要性
        feature_names = FeatureExtractor.get_feature_names()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 儲存結果
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存模型
        model.save_model(save_dir / 'xgboost_model.json')
        
        # 儲存特徵重要性
        feature_importance.to_csv(
            save_dir / 'feature_importance.csv',
            index=False
        )
        
        # 生成視覺化報告
        self._generate_report(
            y_test, y_pred_test, y_prob_test,
            feature_importance, cm,
            save_dir
        )
        
        print(f"\n✓ 模型和報告已儲存至: {save_dir}")
        
        return {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'y_prob_test': y_prob_test,
            'confusion_matrix': cm
        }
    
    def _generate_report(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        y_prob_test: np.ndarray,
        feature_importance: pd.DataFrame,
        cm: np.ndarray,
        save_dir: Path
    ):
        """
        生成視覺化分類報告
        
        Args:
            y_test: 真實標籤
            y_pred_test: 預測標籤
            y_prob_test: 預測機率
            feature_importance: 特徵重要性 DataFrame
            cm: 混淆矩陣
            save_dir: 儲存目錄
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 混淆矩陣
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. ROC 曲線
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall 曲線
        ax3 = plt.subplot(2, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
        ax3.plot(recall, precision)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.grid(True, alpha=0.3)
        
        # 4. 預測機率分佈
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(y_prob_test[y_test == 0], bins=20, alpha=0.5, label='Class 0')
        ax4.hist(y_prob_test[y_test == 1], bins=20, alpha=0.5, label='Class 1')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Count')
        ax4.set_title('Prediction Probability Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Top 20 特徵重要性
        ax5 = plt.subplot(2, 3, 5)
        top_features = feature_importance.head(20)
        ax5.barh(range(len(top_features)), top_features['importance'])
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['feature'], fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 20 Feature Importance')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. 分類指標摘要
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics_text = f"""
Classification Metrics (Test Set)

Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}
Precision: {precision_score(y_test, y_pred_test):.4f}
Recall:    {recall_score(y_test, y_pred_test):.4f}
F1-Score:  {f1_score(y_test, y_pred_test):.4f}
ROC-AUC:   {roc_auc:.4f}

Confusion Matrix:
  TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}
  FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}

Total Samples: {len(y_test)}
  Class 0: {np.sum(y_test == 0)}
  Class 1: {np.sum(y_test == 1)}
        """
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=11, 
                family='monospace', verticalalignment='center')
        ax6.set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'classification_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 視覺化報告已儲存: classification_report.png")