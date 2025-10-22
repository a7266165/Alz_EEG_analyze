"""
EEG 資料載入模組
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class EEGDataset:
    """EEG 資料集封裝"""
    demographics: pd.DataFrame
    metadata: Dict 
    edf_paths: Optional[Dict[str, Path]] = None
    
    def __post_init__(self):
        """驗證資料"""
        if self.demographics.empty:
            warnings.warn("Empty demographics dataframe")
    
    def __len__(self):
        return len(self.demographics)


class DataLoader:
    """
    EEG 資料載入器
    
    統一配置，單一外部介面
    """

    def __init__(
        self,
        demographics_dir: Path,
        eeg_dir: Path,
        groups: List[str] = None,
        cdr_thresholds: List[float] = None,
        age_range: Optional[Tuple[float, float]] = None,
        data_balancing: bool = True,
        use_all_visits: bool = True,
        dataset_selection: str = "all",
        n_bins: int = 5,
        random_seed: int = 42,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        初始化資料載入器
        
        Args:
            demographics_dir: 人口學資料目錄
            eeg_dir: EEG 資料目錄（包含 path.txt）
            groups: 要載入的組別列表
            cdr_thresholds: CDR 閾值列表
            age_range: 年齡範圍 (min, max)
            data_balancing: 是否進行資料平衡
            use_all_visits: 是否使用所有訪視
            dataset_selection: 資料集選擇 ("first", "second", "all")
                - "first": 只載入 P/1
                - "second": 載入 ACS + NAD + P/2
                - "all": 載入全部 (ACS + NAD + P/1 + P/2)
            n_bins: 年齡分箱數量
            random_seed: 隨機種子
            use_cache: 是否使用快取
            cache_dir: 快取目錄
        """
        self.demographics_dir = Path(demographics_dir)
        self.eeg_dir = Path(eeg_dir)
        
        # 配置參數
        self.groups = groups or ["ACS", "NAD", "P"]
        self.cdr_thresholds = cdr_thresholds if cdr_thresholds is not None else [0.5, 1.0, 2.0]
        self.age_range = age_range
        self.data_balancing = data_balancing
        self.use_all_visits = use_all_visits
        self.n_bins = n_bins
        self.random_seed = random_seed
        
        # 資料集選擇驗證
        self.dataset_selection = dataset_selection.lower()
        if self.dataset_selection not in ["first", "second", "all"]:
            raise ValueError(
                f"dataset_selection must be 'first', 'second', or 'all', "
                f"got '{dataset_selection}'"
            )
        
        # 快取設定
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or "workspace/cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 內部快取
        self._demographics_cache = {}
        self._edf_paths_cache = {}
    
    # ========== 主要外部介面 ==========
    
    def load_datasets(self) -> List[EEGDataset]:
        """
        載入所有配置的資料集
        
        Returns:
            資料集列表
        """
        logger.info("開始載入資料集...")
        logger.info(f"資料集選擇: {self.dataset_selection}")
        logger.info(f"組別: {self.groups}")
        logger.info(f"CDR 閾值: {self.cdr_thresholds}")
        logger.info(f"年齡範圍: {self.age_range}")
        logger.info(f"資料平衡: {self.data_balancing}")
        logger.info(f"使用所有訪視: {self.use_all_visits}")
        
        datasets = []
        
        # 為每個 CDR 閾值創建資料集
        for cdr_threshold in self.cdr_thresholds:
            try:
                dataset = self._create_dataset(cdr_threshold)
                datasets.append(dataset)
                
                logger.info(
                    f"✓ 載入資料集: CDR{cdr_threshold} "
                    f"({len(dataset)} 樣本)"
                )
            except Exception as e:
                logger.warning(f"✗ 跳過資料集 CDR{cdr_threshold}: {e}")
        
        logger.info(f"成功載入 {len(datasets)} 個資料集")
        return datasets
    
    # ========== 內部載入方法 ==========
    
    def _create_dataset(self, cdr_threshold: float) -> EEGDataset:
        """創建單一資料集"""
        # 根據 dataset_selection 決定載入哪些組別和 P 的子目錄
        selection_config = {
            "first": (["P"], ["1"]),
            "second": (["ACS", "NAD", "P"], ["2"]),
            "all": (self.groups, ["1", "2"])
        }
        groups_to_load, p_subdirs = selection_config[self.dataset_selection]
        
        # Step 1: 載入人口學資料
        demographics_list = []
        for group in groups_to_load:
            df = self._load_single_demographics(group)
            df['Group'] = group
            demographics_list.append(df)
            logger.debug(f"{group}: {len(df)} 筆, 欄位包含 'ID': {'ID' in df.columns}")
        
        combined_df = pd.concat(demographics_list, ignore_index=True)
        logger.info(f"載入人口學資料: {len(combined_df)} 筆")
        logger.debug(f"合併後欄位: {list(combined_df.columns)[:10]}...")
        logger.debug(f"合併後包含 'ID': {'ID' in combined_df.columns}")
        
        # Step 2: 配對 EDF 路徑，只保留有 EDF 的記錄
        edf_paths = self._match_edf_to_ids_simple(groups_to_load, p_subdirs)
        
        if 'ID' in combined_df.columns:
            original_count = len(combined_df)
            has_edf_mask = combined_df['ID'].isin(edf_paths.keys())
            combined_df = combined_df[has_edf_mask].copy()
            
            logger.info(
                f"EDF 配對篩選: {original_count} → {len(combined_df)} "
                f"({len(combined_df)/original_count*100:.1f}%)"
            )
        else:
            logger.warning("人口學資料中沒有 'ID' 欄位，無法進行 EDF 配對")
        
        # Step 3: 篩選資料（CDR、年齡等）
        filtered_df = self._filter_demographics(
            combined_df,
            cdr_threshold=cdr_threshold
        )
        
        # Step 4: 資料平衡
        if self.data_balancing:
            filtered_df = self._apply_data_balancing(filtered_df)
        
        # Step 5: 最終配對（只配對保留下來的 ID）
        final_edf_paths = {
            subject_id: path 
            for subject_id, path in edf_paths.items() 
            if subject_id in filtered_df['ID'].values
        }
        
        # 建立元資料
        metadata = {
            'groups': groups_to_load,
            'dataset_selection': self.dataset_selection,
            'p_subdirs': p_subdirs,
            'cdr_threshold': cdr_threshold,
            'age_range': self.age_range,
            'data_balancing': self.data_balancing,
            'use_all_visits': self.use_all_visits,
            'n_samples': len(filtered_df),
            'class_distribution': {
                'negative': int((filtered_df['label'] == 0).sum()),
                'positive': int((filtered_df['label'] == 1).sum())
            }
        }
        
        return EEGDataset(
            demographics=filtered_df,
            metadata=metadata,
            edf_paths=final_edf_paths
        )
    
    def _load_single_demographics(self, group: str) -> pd.DataFrame:
        """載入單一組別的人口統計資料（含快取）"""
        # 檢查記憶體快取
        if group in self._demographics_cache:
            logger.debug(f"從記憶體快取載入 {group}")
            return self._demographics_cache[group].copy()
        
        # 檢查檔案快取
        if self.use_cache:
            cache_path = self.cache_dir / f"demographics_{group}.pkl"
            if cache_path.exists():
                logger.debug(f"從快取載入 {group}")
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                    self._demographics_cache[group] = df
                    return df.copy()
        
        # 從 CSV 載入
        csv_path = self.demographics_dir / f"{group}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # 簡單讀取（與 face_analyze 相同）
        df = pd.read_csv(csv_path)
        
        # Debug: 檢查讀取結果
        logger.debug(f"載入 {group}.csv: {len(df)} 筆，欄位: {list(df.columns)[:5]}...")
        
        if 'ID' not in df.columns:
            logger.error(f"{group}.csv 缺少 'ID' 欄位！實際欄位: {list(df.columns)}")
            raise ValueError(f"{group}.csv 缺少必要的 'ID' 欄位")
        
        # 轉換資料型態
        if "Age" in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        if "Sex" in df.columns:
            # 保留原始值，不做轉換（與 face_analyze 不同，因為我們不需要數值編碼）
            pass
        
        if "Global_CDR" in df.columns:
            df['Global_CDR'] = pd.to_numeric(df['Global_CDR'], errors='coerce')
            df['Global_CDR'] = df['Global_CDR'].fillna(0)
        
        # 儲存快取
        if self.use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.debug(f"已快取 {group}")
        
        self._demographics_cache[group] = df
        return df.copy()
    
    def _get_eeg_file_paths(self, group: str, p_subdirs: List[str] = None) -> List[Path]:
        """
        取得單一組別的 EEG 檔案路徑
        
        Args:
            group: 組別名稱
            p_subdirs: P 組的子目錄列表（None 表示全部）
        
        Returns:
            EDF 檔案路徑列表
        """
        if p_subdirs is None:
            p_subdirs = ["1", "2"]
        
        # 檢查快取
        cache_key = f"{group}_{'_'.join(p_subdirs)}"
        if cache_key in self._edf_paths_cache:
            return self._edf_paths_cache[cache_key]
        
        # 讀取 EEG 根目錄
        eeg_path_file = self.eeg_dir / "path.txt"
        if eeg_path_file.exists():
            with open(eeg_path_file, 'r', encoding='utf-8') as f:
                edf_root = Path(f.read().strip())
        else:
            raise FileNotFoundError(f"path.txt not found: {eeg_path_file}")
        
        if not edf_root.exists():
            raise FileNotFoundError(f"EDF directory not found: {edf_root}")
        
        # 掃描檔案
        grp_dir = edf_root / group
        if not grp_dir.exists():
            warnings.warn(f"Group directory not found: {grp_dir}")
            return []
        
        if group == "P":
            edf_files = []
            for subdir in p_subdirs:
                sub_path = grp_dir / subdir
                if sub_path.exists():
                    edf_files.extend(sorted(sub_path.glob("*.edf")))
                else:
                    logger.debug(f"P 子目錄不存在: {sub_path}")
        else:
            edf_files = sorted(grp_dir.glob("*.edf"))
        
        self._edf_paths_cache[cache_key] = edf_files
        return edf_files
    
    # ========== 資料篩選與平衡 ==========
    
    def _filter_demographics(
        self,
        df: pd.DataFrame,
        cdr_threshold: float
    ) -> pd.DataFrame:
        """
        篩選人口學資料
        
        規則：
        1. 基本標籤：ACS/NAD → 0, P → 1
        2. CDR 篩選：NAD (CDR <= threshold), P (CDR >= threshold)
        3. 年齡篩選（如果設定）
        4. 訪視篩選（如果設定）
        """
        df = df.copy()
        
        # 確保 Global_CDR 為數值，空值填充為 0
        if 'Global_CDR' in df.columns:
            df['Global_CDR'] = pd.to_numeric(df['Global_CDR'], errors='coerce')
            df['Global_CDR'] = df['Global_CDR'].fillna(0)
        
        # Step 1: 設定標籤（ACS/NAD → 0, P → 1）
        df['label'] = (df['Group'] == 'P').astype(int)
        
        # Step 2: CDR 篩選（向量化）
        original_count = len(df)
        mask = (
            (df['Group'] == 'ACS') |  # ACS 全保留
            ((df['Group'] == 'NAD') & (df['Global_CDR'] <= cdr_threshold)) |
            ((df['Group'] == 'P') & (df['Global_CDR'] >= cdr_threshold))
        )
        df = df[mask].copy()
        logger.debug(
            f"CDR 篩選: {original_count} → {len(df)} "
            f"({len(df)/original_count*100:.1f}%)"
        )
        
        # Step 3: 年齡篩選
        if self.age_range is not None and 'Age' in df.columns:
            min_age, max_age = self.age_range
            before = len(df)
            df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)].copy()
            logger.debug(
                f"年齡篩選 [{min_age}, {max_age}]: "
                f"{before} → {len(df)} ({len(df)/before*100:.1f}%)"
            )
        
        # Step 4: 訪視篩選
        if not self.use_all_visits and 'ID' in df.columns:
            df = self._keep_latest_visit(df)
            logger.debug(f"保留最新訪視: {len(df)} 筆")
        
        return df
    
    def _keep_latest_visit(self, df: pd.DataFrame) -> pd.DataFrame:
        """保留每個受試者的最新訪視"""
        df = df.copy()
        
        # 從 ID 提取基礎 ID 和訪視號
        df['base_id'] = df['ID'].str.extract(r'^([A-Z]+\d+)', expand=False)
        df['visit'] = pd.to_numeric(
            df['ID'].str.extract(r'-(\d+)$', expand=False),
            errors='coerce'
        ).fillna(1)
        
        # 保留最大訪視號並清理臨時欄位
        return (df.sort_values('visit', ascending=False)
                  .groupby('base_id').first()
                  .reset_index(drop=True)
                  .drop(columns=['base_id', 'visit'], errors='ignore'))
    
    def _apply_data_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        執行資料平衡（按年齡分箱）
        
        策略：在每個年齡箱內，按整體比例抽樣
        """
        if 'label' not in df.columns or 'Age' not in df.columns:
            warnings.warn("缺少 'label' 或 'Age' 欄位，跳過平衡")
            return df
        
        logger.info("執行資料平衡...")
        
        # 分離健康組和病患組
        health_group = df[df['label'] == 0].copy()
        patient_group = df[df['label'] == 1].copy()
        
        n_health = len(health_group)
        n_patient = len(patient_group)
        
        if n_health == 0 or n_patient == 0:
            warnings.warn("某一組為空，跳過平衡")
            return df
        
        # 計算目標比例
        target_ratio = min(n_health, n_patient) / max(n_health, n_patient)
        majority_is_health = n_health > n_patient
        
        logger.debug(
            f"原始樣本數: 健康組 {n_health}, 病患組 {n_patient}, "
            f"目標比例 {target_ratio:.2f}"
        )
        
        # 建立年齡分箱（自動調整箱數）
        all_ages = pd.concat([health_group['Age'], patient_group['Age']])
        n_bins_effective = min(self.n_bins, all_ages.nunique())
        try:
            age_bins = pd.qcut(all_ages, q=n_bins_effective, duplicates='drop', retbins=True)[1]
        except ValueError:
            age_bins = pd.qcut(all_ages, q=2, duplicates='drop', retbins=True)[1]
            logger.debug(f"年齡箱數自動調整為 2")
        
        health_group['age_bin'] = pd.cut(health_group['Age'], bins=age_bins, include_lowest=True)
        patient_group['age_bin'] = pd.cut(patient_group['Age'], bins=age_bins, include_lowest=True)
        
        # 在每個箱中平衡
        balanced_dfs = []
        rng = np.random.RandomState(self.random_seed)
        
        for bin_val in health_group['age_bin'].cat.categories:
            health_in_bin = health_group[health_group['age_bin'] == bin_val]
            patient_in_bin = patient_group[patient_group['age_bin'] == bin_val]
            
            n_health_bin = len(health_in_bin)
            n_patient_bin = len(patient_in_bin)
            
            if n_health_bin == 0 or n_patient_bin == 0:
                if n_health_bin > 0:
                    balanced_dfs.append(health_in_bin)
                if n_patient_bin > 0:
                    balanced_dfs.append(patient_in_bin)
                continue
            
            if majority_is_health:
                target_health = int(n_patient_bin / target_ratio)
                target_health = min(target_health, n_health_bin)
                
                if n_health_bin > target_health:
                    health_sample = health_in_bin.sample(n=target_health, random_state=rng)
                else:
                    health_sample = health_in_bin
                
                balanced_dfs.append(health_sample)
                balanced_dfs.append(patient_in_bin)
            else:
                target_patient = int(n_health_bin / target_ratio)
                target_patient = min(target_patient, n_patient_bin)
                
                if n_patient_bin > target_patient:
                    patient_sample = patient_in_bin.sample(n=target_patient, random_state=rng)
                else:
                    patient_sample = patient_in_bin
                
                balanced_dfs.append(health_in_bin)
                balanced_dfs.append(patient_sample)
        
        if not balanced_dfs:
            warnings.warn("無法進行平衡")
            return df
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        result = result.drop(columns=['age_bin'], errors='ignore')
        
        final_health = len(result[result['label'] == 0])
        final_patient = len(result[result['label'] == 1])
        
        logger.info(
            f"平衡完成: {len(df)} → {len(result)} "
            f"(健康={final_health}, 病患={final_patient})"
        )
        
        return result
    
    def _match_edf_to_ids_simple(
        self,
        groups: List[str],
        p_subdirs: List[str] = None
    ) -> Dict[str, Path]:
        """
        掃描並配對 EDF 檔案（不依賴 DataFrame）
        
        Args:
            groups: 組別列表
            p_subdirs: P 組的子目錄列表
        
        Returns:
            {subject_id: edf_path}
        """
        if p_subdirs is None:
            p_subdirs = ["1", "2"]
        
        edf_dict = {}
        
        for group in groups:
            edf_paths = self._get_eeg_file_paths(group, p_subdirs)
            for edf_path in edf_paths:
                file_id = edf_path.stem
                edf_dict[file_id] = edf_path
        
        logger.debug(f"掃描到 {len(edf_dict)} 個 EDF 檔案")
        return edf_dict