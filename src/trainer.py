"""
EEG 特徵提取與 XGBoost 訓練
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import List, Dict, Tuple, Union
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.analyzer import EEGFeatures, EEGAnalyzer


class FeatureExtractor:
    """
    模組化特徵提取器
    支援 EEG、人口統計、問卷等多種特徵組合
    """
    
    # TODO: 把所有特徵組合寫到一個config裡
    # 預設特徵配置
    FEATURE_CONFIGS = {
        1: ['questionnaire'],                                           # 問卷
        2: ['demographics'],                                            # 年紀+性別
        3: ['eeg_absolute'],                                           # 絕對能量
        4: ['eeg_relative'],                                           # 相對能量
        5: ['eeg_absolute', 'eeg_relative'],                          # 絕對+相對能量
        6: ['eeg_absolute', 'eeg_relative', 'ttest'],                 # t-test能量
        7: ['connectivity'],                                           # 連接性
        8: ['eeg_absolute', 'eeg_relative', 'questionnaire'],         # 能量+問卷
        9: ['eeg_absolute', 'eeg_relative', 'demographics'],          # 能量+人口
        10: ['eeg_absolute', 'eeg_relative', 'connectivity'],         # 能量+連接性
        11: ['eeg_absolute', 'eeg_relative', 'demographics', 'questionnaire'],  # 能量+人口+問卷
        12: ['eeg_absolute', 'eeg_relative', 'connectivity', 'demographics'],   # 能量+連接性+人口
        13: ['eeg_absolute', 'eeg_relative', 'connectivity', 'questionnaire'],  # 能量+連接性+問卷
        14: ['eeg_absolute', 'eeg_relative', 'connectivity', 'demographics', 'questionnaire'],  # 能量+連接性+人口+問卷
        15: ['eeg_absolute', 'eeg_relative', 'ttest', 'questionnaire'],         # t-test能量+問卷
        16: ['eeg_absolute', 'eeg_relative', 'ttest', 'demographics'],          # t-test能量+人口
        17: ['eeg_absolute', 'eeg_relative', 'ttest', 'connectivity'],          # t-test能量+連接性
        18: ['eeg_absolute', 'eeg_relative', 'ttest', 'demographics', 'questionnaire'],  # t-test能量+人口+問卷
        19: ['eeg_absolute', 'eeg_relative', 'ttest', 'connectivity', 'demographics'],   # t-test能量+連接性+人口
        20: ['eeg_absolute', 'eeg_relative', 'ttest', 'connectivity', 'questionnaire'],  # t-test能量+連接性+問卷
        21: ['eeg_absolute', 'eeg_relative', 'ttest', 'connectivity', 'demographics', 'questionnaire']  # t-test能量+連接性+人口+問卷
    }

    def __init__(self, ttest_results: pd.DataFrame = None):
        """
        Args:
            ttest_results: T檢定結果，用於篩選顯著特徵
        """
        self.ttest_results = ttest_results
        self._significant_features = None
        
        if ttest_results is not None:
            self._identify_significant_features()
    
    def _identify_significant_features(self, threshold: float = 0.05):
        """識別 t-test 顯著的特徵"""
        if self.ttest_results is None:
            return
        
        significant_bands = []
        
        for _, row in self.ttest_results.iterrows():
            band = row['Band']
            if row['Absolute_p'] < threshold:
                significant_bands.append((band, 'abs'))
            if row['Relative_p'] < threshold:
                significant_bands.append((band, 'rel'))
        
        self._significant_features = significant_bands
    
    def extract(
        self, 
        features: EEGFeatures,
        demographics: Dict = None,
        feature_components: Union[List[str], str, int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        提取特徵向量
        
        Args:
            features: EEGFeatures 物件
            demographics: 人口統計資料 {'age': float, 'sex': str, 'MMSE': float, ...}
            feature_components: 特徵組件列表
            
        Returns:
            (特徵向量, 特徵名稱列表)
        """
        # 解析特徵配置
        components = self._parse_components(feature_components)
        
        feature_vector = []
        feature_names = []
        
        # 檢查是否需要 t-test 篩選
        use_ttest = 'ttest' in components
        if use_ttest:
            components = [c for c in components if c != 'ttest']
        
        # EEG 絕對能量
        if 'eeg_absolute' in components:
            abs_features, abs_names = self._extract_eeg_features(
                features, 'absolute', use_ttest
            )
            feature_vector.extend(abs_features)
            feature_names.extend(abs_names)
        
        # EEG 相對能量
        if 'eeg_relative' in components:
            rel_features, rel_names = self._extract_eeg_features(
                features, 'relative', use_ttest
            )
            feature_vector.extend(rel_features)
            feature_names.extend(rel_names)
        
        # 頻道間連接性
        if 'connectivity' in components:
            conn_features, conn_names = self._extract_connectivity(features)
            feature_vector.extend(conn_features)
            feature_names.extend(conn_names)
        
        # 人口統計
        if 'demographics' in components:
            if demographics:
                demo_features, demo_names = self._extract_demographics(demographics)
                feature_vector.extend(demo_features)
                feature_names.extend(demo_names)
            else:
                print("警告: 需要人口統計資料但未提供")
        
        # 問卷
        if 'questionnaire' in components:
            if demographics:
                quest_features, quest_names = self._extract_questionnaire(demographics)
                feature_vector.extend(quest_features)
                feature_names.extend(quest_names)
            else:
                print("警告: 需要問卷資料但未提供")
                # 提供預設值避免空特徵
                feature_vector.extend([0.0, 0.0])  # MMSE, CASI
                feature_names.extend(['MMSE', 'CASI'])
        
        if len(feature_vector) == 0:
            raise ValueError(f"警告: 沒有提取到任何特徵，請檢查配置和資料。")

        return np.array(feature_vector), feature_names
    
    def _parse_components(self, components: Union[List[str], str, int]) -> List[str]:
        """解析特徵組件配置"""
        if components is None:
            return ['eeg_absolute', 'eeg_relative']
        
        # 預設配置編號
        if isinstance(components, int):
            if components not in self.FEATURE_CONFIGS:
                raise ValueError(f"Invalid config number: {components}")
            return self.FEATURE_CONFIGS[components].copy()
        
        return list(components)
    
    def _extract_eeg_features(
        self, 
        features: EEGFeatures, 
        power_type: str,
        use_ttest: bool
    ) -> Tuple[List[float], List[str]]:
        """提取 EEG 能量特徵"""
        feature_vector = []
        feature_names = []
        
        is_absolute = (power_type == 'absolute')
        power_data = features.band_powers if is_absolute else features.relative_powers
        suffix = 'abs' if is_absolute else 'rel'
        
        for i, ch in enumerate(features.channels):
            for band in EEGAnalyzer.BANDS.keys():
                # t-test 篩選
                if use_ttest and self._significant_features:
                    if (band, suffix) not in self._significant_features:
                        continue
                
                feature_vector.append(power_data[band][i])
                feature_names.append(f"{ch}_{band}_{suffix}")
        
        return feature_vector, feature_names
    
    def _extract_demographics(self, demographics: Dict) -> Tuple[List[float], List[str]]:
        """提取人口統計特徵"""
        features = []
        names = []
        
        # 年齡
        age_value = demographics.get('age')
        if age_value is not None and pd.notna(age_value):
            try:
                features.append(float(age_value))
                names.append('age')
            except (ValueError, TypeError):
                features.append(0.0)
                names.append('age')
        else:
            features.append(0.0)
            names.append('age')
        
        # 性別 (編碼為 0/1)
        sex_value = demographics.get('sex')
        if sex_value is not None:
            sex_encoded = 1 if sex_value in ['M', 'Male', '男'] else 0
            features.append(float(sex_encoded))
            names.append('sex')
        else:
            features.append(0.0)
            names.append('sex')
        
        return features, names
    
    def _extract_questionnaire(self, demographics: Dict) -> Tuple[List[float], List[str]]:
        """提取問卷特徵"""
        features = []
        names = ['MMSE', 'CASI']  # 固定特徵名稱
        
        # MMSE
        mmse_value = demographics.get('MMSE')
        if mmse_value is not None and pd.notna(mmse_value):
            try:
                features.append(float(mmse_value))
            except (ValueError, TypeError):
                features.append(0.0)
        else:
            features.append(0.0)
        
        # CASI
        casi_value = demographics.get('CASI')
        if casi_value is not None and pd.notna(casi_value):
            try:
                features.append(float(casi_value))
            except (ValueError, TypeError):
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features, names
    
    def _extract_connectivity(self, features: EEGFeatures) -> Tuple[List[float], List[str]]:
        """
        提取頻道間連接性特徵
        
        使用相關係數矩陣的上三角部分（不含對角線）
        
        Args:
            features: EEGFeatures 物件
            
        Returns:
            (特徵向量, 特徵名稱)
        """
        feature_vector = []
        feature_names = []
        
        # 取得相關係數矩陣
        corr_matrix = features.correlation_matrix
        n_channels = len(features.channels)
        
        # 提取上三角部分（不含對角線）
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # 取得頻道名稱（去除可能的前綴）
                ch1 = self._clean_channel_name(features.channels[i])
                ch2 = self._clean_channel_name(features.channels[j])
                
                # 加入相關係數
                feature_vector.append(corr_matrix[i, j])
                feature_names.append(f"{ch1}-{ch2}_corr")
        
        # 可選：加入特定頻帶的連接性
        # 這裡我們計算各頻帶能量的頻道間相關性
        for band_name in EEGAnalyzer.BANDS.keys():
            band_powers = features.band_powers[band_name]
            
            # 計算這個頻帶下不同頻道間的功率相關性
            # 使用簡單的統計量：平均功率差異和變異係數
            mean_power = np.mean(band_powers)
            std_power = np.std(band_powers)
            
            if mean_power > 0:
                cv = std_power / mean_power  # 變異係數
                feature_vector.append(cv)
                feature_names.append(f"{band_name}_cv")
            
            # 前後腦半球的功率比（簡單的不對稱性指標）
            left_channels = [i for i, ch in enumerate(features.channels) 
                           if any(x in ch.upper() for x in ['F3', 'C3', 'P3', 'T3', 'O1'])]
            right_channels = [i for i, ch in enumerate(features.channels) 
                            if any(x in ch.upper() for x in ['F4', 'C4', 'P4', 'T4', 'O2'])]
            
            if left_channels and right_channels:
                left_power = np.mean([band_powers[i] for i in left_channels])
                right_power = np.mean([band_powers[i] for i in right_channels])
                
                if (left_power + right_power) > 0:
                    asymmetry = (left_power - right_power) / (left_power + right_power)
                    feature_vector.append(asymmetry)
                    feature_names.append(f"{band_name}_asymmetry")
        
        return feature_vector, feature_names
    
    def _clean_channel_name(self, channel: str) -> str:
        """清理頻道名稱，去除前綴和後綴"""
        # 處理 "EEG F3-REF" 格式
        if 'EEG' in channel and '-REF' in channel:
            return channel.replace('EEG ', '').replace('-REF', '')
        return channel


class XGBoostTrainer:
    """
    XGBoost 訓練器
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        feature_components: Union[List[str], str, int] = None,
        ttest_results: pd.DataFrame = None,
        **xgb_params
    ):
        """
        Args:
            test_size: 測試集比例
            random_state: 隨機種子
            feature_components: 特徵組件配置
            ttest_results: T檢定結果（用於特徵篩選）
            **xgb_params: XGBoost 參數
        """
        self.test_size = test_size
        self.random_state = random_state
        self.feature_components = feature_components
        
        # 初始化特徵提取器
        self.extractor = FeatureExtractor(ttest_results)
        
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
        labels: List[int],
        demographics_list: List[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        準備訓練數據
        
        Args:
            features_list: EEGFeatures 列表
            labels: 標籤列表
            demographics_list: 人口統計資料列表
            
        Returns:
            (X, y, subject_ids, feature_names) 特徵矩陣、標籤、ID、特徵名稱
        """
        X = []
        subject_ids = []
        
        for i, features in enumerate(features_list):
            demographics = demographics_list[i] if demographics_list else None
            feature_vector, feature_names = self.extractor.extract(
                features, 
                demographics,
                self.feature_components
            )
            X.append(feature_vector)
            subject_ids.append(features.subject_id)
        
        X = np.array(X)
        y = np.array(labels)
        
        return X, y, subject_ids, feature_names
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: List[str],
        feature_names: List[str] = None,
        save_dir: Path = Path("workspace/analyze_result")
    ) -> Dict:
        """
        訓練 XGBoost 模型（確保同一個人的資料在同一集）
        
        Args:
            X: 特徵矩陣
            y: 標籤向量
            subject_ids: 受試者ID列表
            feature_names: 特徵名稱列表
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