"""
EEG 分析主程式
包含特徵萃取、快取、統計報告和機器學習訓練
"""
import sys
from pathlib import Path
import pickle
import hashlib
import json
import pandas as pd
import numpy as np

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer import EEGAnalyzer
from src.trainer import XGBoostTrainer
from src.loader import DataLoader


class FeatureCache:
    """特徵快取管理器"""
    
    def __init__(self, cache_dir: Path = Path("workspace/features")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, dataset_metadata: dict) -> str:
        """
        根據資料集參數生成唯一的快取鍵
        
        Args:
            dataset_metadata: 資料集元數據
            
        Returns:
            快取鍵（hash）
        """
        # 使用關鍵參數生成 hash
        key_params = {
            'groups': sorted(dataset_metadata.get('groups', [])),
            'cdr_threshold': dataset_metadata.get('cdr_threshold'),
            'dataset_selection': dataset_metadata.get('dataset_selection'),
        }
        key_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cache_path(self, dataset_metadata: dict) -> Path:
        """取得快取檔案路徑"""
        cache_key = self._get_cache_key(dataset_metadata)
        return self.cache_dir / f"features_{cache_key}.pkl"
    
    def exists(self, dataset_metadata: dict) -> bool:
        """檢查快取是否存在"""
        cache_path = self.get_cache_path(dataset_metadata)
        return cache_path.exists()
    
    def save(self, dataset_metadata: dict, features_list: list, labels: list, subject_ids: list):
        """
        儲存特徵到快取
        
        Args:
            dataset_metadata: 資料集元數據
            features_list: 特徵列表
            labels: 標籤列表
            subject_ids: 受試者ID列表
        """
        cache_path = self.get_cache_path(dataset_metadata)
        
        cache_data = {
            'features_list': features_list,
            'labels': labels,
            'subject_ids': subject_ids,
            'metadata': dataset_metadata,
            'n_samples': len(features_list)
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✓ 特徵已快取至: {cache_path.name}")
    
    def load(self, dataset_metadata: dict) -> tuple:
        """
        從快取載入特徵
        
        Returns:
            (features_list, labels, subject_ids)
        """
        cache_path = self.get_cache_path(dataset_metadata)
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"✓ 從快取載入特徵: {cache_path.name}")
        print(f"  樣本數: {cache_data['n_samples']}")
        
        return (
            cache_data['features_list'],
            cache_data['labels'],
            cache_data['subject_ids']
        )


def extract_or_load_features(
    dataset,
    analyzer: EEGAnalyzer,
    cache: FeatureCache,
    force_recompute: bool = False
) -> tuple:
    """
    萃取特徵或從快取載入
    
    Args:
        dataset: 資料集物件
        analyzer: EEG 分析器
        cache: 快取管理器
        force_recompute: 是否強制重新計算
        
    Returns:
        (features_list, labels, subject_ids)
    """
    # 檢查快取
    if not force_recompute and cache.exists(dataset.metadata):
        print("\n發現快取，直接載入...")
        return cache.load(dataset.metadata)
    
    # 快取不存在，進行分析
    print("\n快取不存在，開始分析 EEG 訊號...")
    features_list = []
    labels = []
    subject_ids_list = []
    
    total = len(dataset.demographics)
    for idx, (_, row) in enumerate(dataset.demographics.iterrows(), 1):
        subject_id = row['ID']
        label = row['label']
        
        if subject_id in dataset.edf_paths:
            path = dataset.edf_paths[subject_id]
            try:
                features = analyzer.analyze_file(path)
                features_list.append(features)
                labels.append(label)
                subject_ids_list.append(subject_id)
                
                if idx % 10 == 0 or idx == total:
                    print(f"  進度: {idx}/{total} ({idx/total*100:.1f}%)")
                    
            except Exception as e:
                print(f"✗ {subject_id}: {e}")
                continue
    
    # 儲存到快取
    cache.save(dataset.metadata, features_list, labels, subject_ids_list)
    
    return features_list, labels, subject_ids_list


def generate_statistical_report(
    analyzer: EEGAnalyzer,
    features_list: list,
    labels: list,
    output_dir: Path,
    generate_topomap: bool = True
):
    """
    生成統計報告（相關係數矩陣、能量分佈、T檢定、頭譜圖）
    
    使用整合後的 analyzer.generate_group_report
    
    Args:
        analyzer: EEG 分析器（整合了報告功能）
        features_list: 特徵列表
        labels: 標籤列表
        output_dir: 輸出目錄
        generate_topomap: 是否生成頭譜圖
    """
    print("\n=== 生成統計報告 ===")
    
    # 根據標籤分組
    control_features = [f for f, l in zip(features_list, labels) if l == 0]
    patient_features = [f for f, l in zip(features_list, labels) if l == 1]
    
    print(f"控制組: {len(control_features)} 樣本")
    print(f"病患組: {len(patient_features)} 樣本")
    
    if len(control_features) == 0 or len(patient_features) == 0:
        print("⚠ 警告：某組樣本數為 0，跳過統計報告")
        return
    
    # 使用整合後的 analyzer 生成報告
    ttest_results = analyzer.generate_group_report(
        group1_features=control_features,
        group2_features=patient_features,
        group1_name="Control",
        group2_name="Patient",
        output_dir=output_dir,
        generate_topomap=generate_topomap  # 新增：啟用頭譜圖
    )
    
    # 顯示顯著性結果摘要
    if ttest_results is not None:
        print("\n=== T檢定結果摘要 ===")
        significant_bands = ttest_results[
            (ttest_results['Absolute_sig'] != 'ns') | 
            (ttest_results['Relative_sig'] != 'ns')
        ]
        if not significant_bands.empty:
            print(significant_bands[['Band', 'Absolute_sig', 'Relative_sig']].to_string(index=False))
        else:
            print("沒有發現顯著差異的頻帶")
    
    return ttest_results


def train_classification_model(
    features_list: list,
    labels: list,
    subject_ids: list,
    output_dir: Path,
    feature_type: str = 'all',
    xgb_params: dict = None,
    n_runs: int = 1,
    base_seed: int = 42
):
    """
    訓練 XGBoost 分類模型
    
    Args:
        features_list: 特徵列表
        labels: 標籤列表
        subject_ids: 受試者ID列表
        output_dir: 輸出目錄
        feature_type: 特徵類型 ('absolute', 'relative', 'all')
        xgb_params: XGBoost 參數
        n_runs: 訓練模型數量
        base_seed: 起始隨機種子
    """
    print("\n=== 訓練分類模型 ===")
    print(f"使用特徵: {feature_type}")
    print(f"訓練次數: {n_runs}")

    all_results = []
    
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n--- 第 {run_idx+1}/{n_runs} 次訓練 (seed={seed}) ---")

        # 預設參數
        default_params = {
            'test_size': 0.2,
            'random_state': seed,
            'n_estimators': 100,
            'max_depth': 5,
            'feature_type': feature_type
        }
        
        if xgb_params:
            default_params.update(xgb_params)
        
        # 訓練
        trainer = XGBoostTrainer(**default_params)
        X, y, subject_ids = trainer.prepare_data(features_list, labels)
        
        # 為每次運行建立子目錄
        run_dir = output_dir / f"run_{run_idx+1}_seed_{seed}"
        results = trainer.train(X, y, subject_ids, save_dir=run_dir)

        all_results.append({
                    'run': run_idx + 1,
                    'seed': seed,
                    'train_acc': results['train_acc'],
                    'test_acc': results['test_acc'],
                    'model': results['model'],
                    'feature_importance': results['feature_importance']
                })
    
    # 統計分析
    if n_runs > 1:
        _analyze_multiple_runs(all_results, output_dir, feature_type)
    else:
        # 單次運行時顯示特徵重要性
        print("\n=== Top 10 重要特徵 ===")
        print(all_results[0]['feature_importance'].head(10).to_string(index=False))
    
    return results

def _analyze_multiple_runs(results: list, output_dir: Path, feature_type: str):
    """
    分析多次運行的結果
    
    Args:
        results: 所有運行結果列表
        output_dir: 輸出目錄
        feature_type: 特徵類型
    """
    import matplotlib.pyplot as plt
    
    # 收集準確率
    train_accs = [r['train_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    # 統計
    train_mean, train_std = np.mean(train_accs), np.std(train_accs)
    test_mean, test_std = np.mean(test_accs), np.std(test_accs)
    
    print("\n" + "=" * 60)
    print(f"多次運行統計 ({len(results)} 次, 特徵類型: {feature_type})")
    print("=" * 60)
    print(f"訓練集準確率: {train_mean:.4f} ± {train_std:.4f}")
    print(f"測試集準確率: {test_mean:.4f} ± {test_std:.4f}")
    print(f"最佳測試準確率: {max(test_accs):.4f} (seed={results[np.argmax(test_accs)]['seed']})")
    print(f"最差測試準確率: {min(test_accs):.4f} (seed={results[np.argmin(test_accs)]['seed']})")
    
    # 彙總特徵重要性
    all_importances = pd.concat([
        r['feature_importance'].assign(run=r['run']) 
        for r in results
    ])
    
    # 計算平均重要性
    avg_importance = (all_importances.groupby('feature')['importance']
                      .agg(['mean', 'std'])
                      .sort_values('mean', ascending=False)
                      .reset_index())
    
    print("\n=== Top 10 平均特徵重要性 ===")
    print(avg_importance.head(10).to_string(index=False))
    
    # 儲存統計結果
    summary = pd.DataFrame(results)[['run', 'seed', 'train_acc', 'test_acc']]
    summary.to_csv(output_dir / 'multiple_runs_summary.csv', index=False)
    avg_importance.to_csv(output_dir / 'average_feature_importance.csv', index=False)
    
    # 繪製準確率分佈圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 準確率箱型圖
    ax1.boxplot([train_accs, test_accs], tick_labels=['Train', 'Test'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Accuracy Distribution ({len(results)} runs)')
    ax1.grid(True, alpha=0.3)
    
    # 準確率趨勢圖
    runs = list(range(1, len(results) + 1))
    ax2.plot(runs, train_accs, 'o-', label=f'Train ({train_mean:.3f}±{train_std:.3f})')
    ax2.plot(runs, test_accs, 's-', label=f'Test ({test_mean:.3f}±{test_std:.3f})')
    ax2.axhline(train_mean, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(test_mean, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Run')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Across Runs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multiple_runs_analysis.png', dpi=300)
    plt.close()
    
    print(f"\n✓ 多次運行分析已儲存至: {output_dir}")


def main_analysis_pipeline(
    groups: list = None,
    cdr_threshold: float = 0.5,
    dataset_selection: str = "all",
    feature_type: str = 'all',
    n_runs: int = 1, 
    base_seed: int = 42,
    force_recompute: bool = False,
    skip_report: bool = False,
    skip_training: bool = False,
    xgb_params: dict = None
):
    """
    主分析流程
    
    Args:
        groups: 要分析的組別
        cdr_threshold: CDR 閾值
        dataset_selection: 資料集選擇 ("first", "second", "all")
        feature_type: 特徵類型 ('absolute', 'relative', 'all')
        force_recompute: 是否強制重新計算特徵
        skip_report: 是否跳過統計報告
        skip_training: 是否跳過模型訓練
        xgb_params: XGBoost 參數
    """
    print("=" * 70)
    print("EEG 阿茲海默症分析流程")
    print("=" * 70)
    print(f"特徵選擇: {feature_type}")
    
    # 設定
    if groups is None:
        groups = ["ACS", "NAD", "P"]
    
    output_dir = Path("workspace/analyze_result")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 載入資料集
    print("\n[步驟 1/4] 載入資料集")
    loader = DataLoader(
        demographics_dir=Path("data/demographics"),
        eeg_dir=Path("data/EEG"),
        groups=groups,
        cdr_thresholds=[cdr_threshold],
        dataset_selection=dataset_selection,
        use_cache=True
    )
    
    datasets = loader.load_datasets()
    
    if len(datasets) == 0:
        print("✗ 沒有載入任何資料集")
        return
    
    # 2. 萃取特徵（或從快取載入）
    print(f"\n[步驟 2/4] 特徵萃取")
    analyzer = EEGAnalyzer(resample_freq=250.0, n_jobs=-1)
    cache = FeatureCache()
    
    results_list = []
    
    for dataset in datasets:
        print(f"\n處理資料集: CDR {dataset.metadata['cdr_threshold']}")
        
        # 萃取或載入特徵
        features_list, labels, subject_ids = extract_or_load_features(
            dataset, analyzer, cache, force_recompute
        )
        
        print(f"\n收集特徵: {len(features_list)} 個樣本")
        
        results_list.append({
            'features_list': features_list,
            'labels': labels,
            'subject_ids': subject_ids,
            'metadata': dataset.metadata
        })
    
    # 3. 生成統計報告
    if not skip_report:
        print(f"\n[步驟 3/4] 生成統計報告")
        for result in results_list:
            # 傳入 analyzer 使用整合後的報告功能
            generate_statistical_report(
                analyzer,  # 新增：傳入 analyzer
                result['features_list'],
                result['labels'],
                output_dir
            )
    else:
        print(f"\n[步驟 3/4] 跳過統計報告")
    
    # 4. 訓練模型
    if not skip_training:
        print(f"\n[步驟 4/4] 訓練分類模型")
        for result in results_list:
            train_classification_model(
                result['features_list'],
                result['labels'],
                result['subject_ids'],
                output_dir,
                feature_type,
                xgb_params,
                n_runs,
                base_seed
            )
    else:
        print(f"\n[步驟 4/4] 跳過模型訓練")
    
    print("\n" + "=" * 70)
    print("✓ 分析完成！")
    print(f"✓ 結果儲存於: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    # 可以根據需求調整參數
    main_analysis_pipeline(
        groups=["ACS", "NAD", "P"],
        cdr_threshold=0.5,
        dataset_selection="second",
        n_runs=100,
        base_seed=42,
        feature_type='absolute',         # 'absolute', 'relative', or 'all'
        force_recompute=False,      # 使用快取（如果存在）
        skip_report=False,          # 生成統計報告
        skip_training=False,        # 訓練分類模型
        xgb_params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        }
    )