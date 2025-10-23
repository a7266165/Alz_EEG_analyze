"""
EEG 頻域分析與報告生成器
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.signal import welch


@dataclass
class EEGFeatures:
    """EEG 特徵結果"""
    subject_id: str
    band_powers: Dict[str, np.ndarray]  # {band: [ch1, ch2, ...]}
    relative_powers: Dict[str, np.ndarray]
    correlation_matrix: np.ndarray
    channels: List[str]


class EEGAnalyzer:
    """
    EEG 頻域分析與報告生成器
    
    分析指定頻帶的能量與頻道相關性，並生成統計報告
    """
    
    # 頻帶定義
    BANDS = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta1': (12, 18),
        'Beta2': (18, 30),
        'Beta3': (30, 40),
        'Gamma': (40, 50)
    }
    
    # 預設頻道
    DEFAULT_CHANNELS = ['F3', 'F4', 'F7', 'F8', 'Fz', 
                        'C3', 'C4', 'Cz', 
                        'P3', 'P4', 'Pz', 
                        'T3', 'T4', 'T5', 'T6', 
                        'O1', 'O2']
    
    def __init__(
        self, 
        resample_freq: float = 250.0, 
        n_jobs: int = -1,
        channels: List[str] = None
    ):
        """
        Args:
            resample_freq: 重採樣頻率 (Hz)
            n_jobs: 並行處理的工作數 (-1 表示使用所有核心)
            channels: 要分析的頻道列表 (None 表示使用預設頻道)
        """
        self.resample_freq = resample_freq
        self.n_jobs = n_jobs
        self.channels = channels or self.DEFAULT_CHANNELS
    
    # ========== 核心分析方法 ==========
    
    def analyze_file(self, edf_path: Path) -> EEGFeatures:
        """
        分析單一 EDF 檔案
        
        Args:
            edf_path: EDF 檔案路徑
            
        Returns:
            EEGFeatures
        """
        # 讀取 EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # 選擇頻道
        available_channels = self._match_channels(raw.ch_names)
        
        # 檢查是否有可用頻道
        if not available_channels:
            raise ValueError(
                f"找不到任何匹配的頻道。\n"
                f"檔案中的頻道: {raw.ch_names}\n"
                f"期望的頻道: {self.channels}"
            )
        
        raw.pick(available_channels)
        
        # 重採樣
        if raw.info['sfreq'] != self.resample_freq:
            raw.resample(self.resample_freq)
        
        # 取得數據
        data = raw.get_data()  # shape: (n_channels, n_times)
        
        # 計算頻帶能量
        band_powers = self._compute_band_powers(data, raw.info['sfreq'])
        
        # 計算相對能量
        relative_powers = self._compute_relative_powers(band_powers)
        
        # 計算相關係數
        correlation_matrix = np.corrcoef(data)
        
        return EEGFeatures(
            subject_id=edf_path.stem,
            band_powers=band_powers,
            relative_powers=relative_powers,
            correlation_matrix=correlation_matrix,
            channels=available_channels
        )
    
    def analyze_batch(self, edf_paths: List[Path]) -> List[EEGFeatures]:
        """
        批次分析多個 EDF 檔案（並行處理）
        
        Args:
            edf_paths: EDF 檔案路徑列表
            
        Returns:
            EEGFeatures 列表
        """
        from joblib import Parallel, delayed
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._safe_analyze)(path) for path in edf_paths
        )
        
        # 過濾失敗的結果
        return [r for r in results if r is not None]
    
    def analyze_dataset(self, edf_paths: Dict[str, Path]):
        """
        逐一分析資料集（生成器模式，節省記憶體）
        
        Args:
            edf_paths: {subject_id: edf_path} 字典
            
        Yields:
            EEGFeatures
        """
        for subject_id, path in edf_paths.items():
            try:
                yield self.analyze_file(path)
            except Exception as e:
                print(f"✗ {subject_id}: {e}")
                continue
    
    # ========== 報告生成方法 ==========
    
    def generate_group_report(
        self,
        group1_features: List[EEGFeatures],
        group2_features: List[EEGFeatures],
        group1_name: str = "Control",
        group2_name: str = "Patient",
        output_dir: Path = Path("workspace/analyze_result"),
        generate_topomap: bool = True
    ):
        """
        生成群組比較報告
        
        Args:
            group1_features: 組別1的特徵列表
            group2_features: 組別2的特徵列表
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            output_dir: 報告輸出目錄
            generate_topomap: 是否生成頭譜圖
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n生成報告...")
        print(f"{group1_name}: {len(group1_features)} 樣本")
        print(f"{group2_name}: {len(group2_features)} 樣本")
        
        # 1. 相關係數矩陣
        self._plot_correlation_matrices(
            group1_features, group2_features,
            group1_name, group2_name,
            output_dir
        )
        
        # 2. 能量分佈與T檢定
        ttest_results = self._analyze_power_distributions(
            group1_features, group2_features,
            group1_name, group2_name,
            output_dir
        )
        
        # 3. 頭譜圖（新增）
        if generate_topomap:
            self._plot_topomaps(
                group1_features, group2_features,
                group1_name, group2_name,
                output_dir
            )
        
        # 4. 儲存統計結果
        ttest_results.to_csv(output_dir / 'ttest_results.csv', index=False)
        
        print(f"\n✓ 報告已儲存至: {output_dir}")
        print(f"  - correlation_matrices.png")
        print(f"  - power_distributions.png")
        if generate_topomap:
            print(f"  - topomap_absolute.png")
            print(f"  - topomap_relative.png")
            print(f"  - topomap_difference.png")
        print(f"  - ttest_results.csv")
        
        return ttest_results
    
    def generate_subject_report(
        self,
        features: EEGFeatures,
        output_path: Path = None
    ) -> pd.DataFrame:
        """
        生成單一受試者報告
        
        Args:
            features: EEGFeatures 物件
            output_path: 儲存路徑 (optional)
            
        Returns:
            特徵 DataFrame
        """
        df = self.to_dataframe(features)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✓ 個人報告已儲存: {output_path}")
        
        return df
    
    def to_dataframe(self, features: EEGFeatures) -> pd.DataFrame:
        """
        將特徵轉換為 DataFrame
        
        Args:
            features: EEGFeatures 物件
            
        Returns:
            特徵 DataFrame
        """
        rows = []
        
        for i, ch in enumerate(features.channels):
            row = {'subject_id': features.subject_id, 'channel': ch}
            
            # 絕對能量
            for band in self.BANDS.keys():
                row[f'{band}_abs'] = features.band_powers[band][i]
            
            # 相對能量
            for band in self.BANDS.keys():
                row[f'{band}_rel'] = features.relative_powers[band][i]
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    # ========== 內部分析方法 ==========
    
    def _safe_analyze(self, path: Path) -> EEGFeatures:
        """
        安全地分析檔案（用於並行處理）
        
        Args:
            path: EDF 檔案路徑
            
        Returns:
            EEGFeatures 或 None（失敗時）
        """
        try:
            return self.analyze_file(path)
        except Exception as e:
            print(f"✗ {path.stem}: {e}")
            return None
    
    def _match_channels(self, available: List[str]) -> List[str]:
        """
        匹配可用頻道（支援兩種格式）
        
        格式1: F3, F4, Cz, ... (36.3%)
        格式2: EEG F3-REF, EEG F4-REF, EEG Cz-REF, ... (63.7%)
        
        Args:
            available: EDF 中的頻道名稱
            
        Returns:
            匹配到的頻道列表
        """
        matched = []
        
        for target in self.channels:
            target_upper = target.upper()
            
            # 嘗試匹配的模式（優先順序）
            patterns = [
                target,                          # 原始名稱 (F3)
                target_upper,                    # 大寫 (F3)
                target.lower(),                  # 小寫 (f3)
                f"EEG {target}-REF",            # 格式2 (EEG F3-REF)
                f"EEG {target_upper}-REF",      # 格式2大寫 (EEG F3-REF)
                f"EEG {target.lower()}-REF",    # 格式2小寫 (EEG f3-REF)
            ]
            
            # 找到第一個匹配的
            for pattern in patterns:
                if pattern in available:
                    matched.append(pattern)
                    break
        
        return matched
    
    def _compute_band_powers(
        self, 
        data: np.ndarray, 
        sfreq: float
    ) -> Dict[str, np.ndarray]:
        """
        計算各頻帶的絕對能量
        
        Args:
            data: 原始信號 (n_channels, n_times)
            sfreq: 採樣率
            
        Returns:
            {band_name: power_array}
        """
        band_powers = {}
        
        for band_name, (fmin, fmax) in self.BANDS.items():
            powers = []
            
            for ch_data in data:
                # Welch 方法計算功率譜密度
                freqs, psd = welch(ch_data, sfreq, nperseg=int(sfreq*2))
                
                # 提取頻帶範圍
                freq_mask = (freqs >= fmin) & (freqs < fmax)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                
                powers.append(band_power)
            
            band_powers[band_name] = np.array(powers)
        
        return band_powers
    
    def _compute_relative_powers(
        self, 
        band_powers: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        計算各頻帶的相對能量
        
        Args:
            band_powers: 絕對能量字典
            
        Returns:
            {band_name: relative_power_array}
        """
        # 計算總能量
        total_power = sum(band_powers.values())
        
        # 計算相對能量
        relative_powers = {
            band: power / total_power 
            for band, power in band_powers.items()
        }
        
        return relative_powers
    
    # ========== 內部報告方法 ==========
    
    def _plot_correlation_matrices(
        self,
        group1_features: List[EEGFeatures],
        group2_features: List[EEGFeatures],
        group1_name: str,
        group2_name: str,
        output_dir: Path
    ):
        """
        繪製相關係數矩陣
        
        Args:
            group1_features: 組別1特徵列表
            group2_features: 組別2特徵列表
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            output_dir: 輸出目錄
        """
        # 計算平均相關係數矩陣
        corr1 = np.mean([f.correlation_matrix for f in group1_features], axis=0)
        corr2 = np.mean([f.correlation_matrix for f in group2_features], axis=0)
        
        # 取得頻道名稱
        channels = group1_features[0].channels
        
        # 繪製
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 組別1
        sns.heatmap(corr1, ax=axes[0], cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, square=True, 
                    xticklabels=channels, yticklabels=channels,
                    cbar_kws={'label': 'Correlation'})
        axes[0].set_title(f'{group1_name} (n={len(group1_features)})')
        
        # 組別2
        sns.heatmap(corr2, ax=axes[1], cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, square=True,
                    xticklabels=channels, yticklabels=channels,
                    cbar_kws={'label': 'Correlation'})
        axes[1].set_title(f'{group2_name} (n={len(group2_features)})')
        
        # 差異
        diff = corr2 - corr1
        sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0,
                    vmin=-0.5, vmax=0.5, square=True,
                    xticklabels=channels, yticklabels=channels,
                    cbar_kws={'label': 'Difference'})
        axes[2].set_title(f'{group2_name} - {group1_name}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrices.png', dpi=300)
        plt.close()
        
        print(f"✓ 相關係數矩陣已儲存")
    
    def _analyze_power_distributions(
        self,
        group1_features: List[EEGFeatures],
        group2_features: List[EEGFeatures],
        group1_name: str,
        group2_name: str,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        分析能量分佈並執行T檢定
        
        Args:
            group1_features: 組別1特徵列表
            group2_features: 組別2特徵列表
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            output_dir: 輸出目錄
            
        Returns:
            T檢定結果 DataFrame
        """
        bands = list(self.BANDS.keys())
        
        # 收集數據
        group1_abs, group1_rel = self._collect_powers(group1_features)
        group2_abs, group2_rel = self._collect_powers(group2_features)
        
        # T檢定
        ttest_results = []
        for band in bands:
            # 絕對能量T檢定
            t_abs, p_abs = stats.ttest_ind(
                group1_abs[band], 
                group2_abs[band]
            )
            
            # 相對能量T檢定
            t_rel, p_rel = stats.ttest_ind(
                group1_rel[band], 
                group2_rel[band]
            )
            
            ttest_results.append({
                'Band': band,
                'Absolute_t': t_abs,
                'Absolute_p': p_abs,
                'Absolute_sig': '***' if p_abs < 0.001 else '**' if p_abs < 0.01 else '*' if p_abs < 0.05 else 'ns',
                'Relative_t': t_rel,
                'Relative_p': p_rel,
                'Relative_sig': '***' if p_rel < 0.001 else '**' if p_rel < 0.01 else '*' if p_rel < 0.05 else 'ns'
            })
        
        # 繪製分佈圖
        self._plot_power_distributions(
            group1_abs, group1_rel,
            group2_abs, group2_rel,
            group1_name, group2_name,
            bands, output_dir
        )
        
        print(f"✓ 能量分佈圖已儲存")
        
        return pd.DataFrame(ttest_results)
    
    def _collect_powers(
        self,
        features_list: List[EEGFeatures]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        收集所有樣本的能量數據
        
        Args:
            features_list: EEGFeatures 列表
            
        Returns:
            (絕對能量字典, 相對能量字典)
        """
        bands = list(self.BANDS.keys())
        
        abs_powers = {band: [] for band in bands}
        rel_powers = {band: [] for band in bands}
        
        for features in features_list:
            for band in bands:
                # 平均所有頻道的能量
                abs_powers[band].append(np.mean(features.band_powers[band]))
                rel_powers[band].append(np.mean(features.relative_powers[band]))
        
        # 轉換為 numpy array
        abs_powers = {band: np.array(vals) for band, vals in abs_powers.items()}
        rel_powers = {band: np.array(vals) for band, vals in rel_powers.items()}
        
        return abs_powers, rel_powers
    
    def _plot_power_distributions(
        self,
        group1_abs: Dict[str, np.ndarray],
        group1_rel: Dict[str, np.ndarray],
        group2_abs: Dict[str, np.ndarray],
        group2_rel: Dict[str, np.ndarray],
        group1_name: str,
        group2_name: str,
        bands: List[str],
        output_dir: Path
    ):
        """
        繪製能量分佈圖
        
        Args:
            group1_abs: 組別1絕對能量
            group1_rel: 組別1相對能量
            group2_abs: 組別2絕對能量
            group2_rel: 組別2相對能量
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            bands: 頻帶名稱列表
            output_dir: 輸出目錄
        """
        fig, axes = plt.subplots(2, len(bands), figsize=(20, 8))
        
        for i, band in enumerate(bands):
            # 絕對能量
            axes[0, i].violinplot(
                [group1_abs[band], group2_abs[band]],
                positions=[1, 2],
                showmeans=True
            )
            axes[0, i].set_xticks([1, 2])
            axes[0, i].set_xticklabels([group1_name, group2_name])
            axes[0, i].set_title(f'{band}')
            axes[0, i].set_ylabel('Absolute Power')
            
            # 相對能量
            axes[1, i].violinplot(
                [group1_rel[band], group2_rel[band]],
                positions=[1, 2],
                showmeans=True
            )
            axes[1, i].set_xticks([1, 2])
            axes[1, i].set_xticklabels([group1_name, group2_name])
            axes[1, i].set_ylabel('Relative Power')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'power_distributions.png', dpi=300)
        plt.close()
    
    def _plot_topomaps(
        self,
        group1_features: List[EEGFeatures],
        group2_features: List[EEGFeatures],
        group1_name: str,
        group2_name: str,
        output_dir: Path
    ):
        """
        繪製群體頭譜圖
        
        Args:
            group1_features: 組別1特徵列表
            group2_features: 組別2特徵列表
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            output_dir: 輸出目錄
        """
        print("✓ 生成頭譜圖...")
        
        # 取得標準電極位置
        montage = self._get_standard_montage()
        
        # 計算群體平均能量
        group1_abs_avg, group1_rel_avg = self._compute_group_average_powers(group1_features)
        group2_abs_avg, group2_rel_avg = self._compute_group_average_powers(group2_features)
        
        # 1. 繪製絕對能量頭譜圖
        self._plot_topomap_grid(
            group1_abs_avg, group2_abs_avg,
            montage, group1_name, group2_name,
            output_dir / 'topomap_absolute.png',
            'Absolute Power',
            use_log=False
        )
        
        # 2. 繪製相對能量頭譜圖
        self._plot_topomap_grid(
            group1_rel_avg, group2_rel_avg,
            montage, group1_name, group2_name,
            output_dir / 'topomap_relative.png',
            'Relative Power',
            use_log=False
        )
        
        # 3. 繪製差異頭譜圖
        self._plot_topomap_difference(
            group1_abs_avg, group2_abs_avg,
            group1_rel_avg, group2_rel_avg,
            montage, group1_name, group2_name,
            output_dir / 'topomap_difference.png'
        )
        
        print(f"✓ 頭譜圖已儲存")
    
    def _get_standard_montage(self):
        """
        取得標準10-20系統電極位置
        
        Returns:
            mne.channels.DigMontage
        """
        # 創建標準10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        
        # 確保頻道名稱匹配
        # 將頻道名稱標準化（去除可能的前綴）
        ch_names_cleaned = []
        for ch in self.channels:
            # 處理 "EEG F3-REF" 格式
            if 'EEG' in ch and '-REF' in ch:
                ch_clean = ch.replace('EEG ', '').replace('-REF', '')
            else:
                ch_clean = ch
            ch_names_cleaned.append(ch_clean.upper())
        
        return montage, ch_names_cleaned
    
    def _compute_group_average_powers(
        self, 
        features_list: List[EEGFeatures]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        計算群體平均能量（每個頻道）
        
        Args:
            features_list: EEGFeatures 列表
            
        Returns:
            (絕對能量平均, 相對能量平均)
        """
        bands = list(self.BANDS.keys())
        
        # 初始化累加器
        abs_sum = {band: None for band in bands}
        rel_sum = {band: None for band in bands}
        
        for features in features_list:
            for band in bands:
                if abs_sum[band] is None:
                    abs_sum[band] = features.band_powers[band].copy()
                    rel_sum[band] = features.relative_powers[band].copy()
                else:
                    abs_sum[band] += features.band_powers[band]
                    rel_sum[band] += features.relative_powers[band]
        
        # 計算平均
        n = len(features_list)
        abs_avg = {band: powers / n for band, powers in abs_sum.items()}
        rel_avg = {band: powers / n for band, powers in rel_sum.items()}
        
        return abs_avg, rel_avg
    
    def _plot_topomap_grid(
        self,
        group1_powers: Dict[str, np.ndarray],
        group2_powers: Dict[str, np.ndarray],
        montage_info: Tuple,
        group1_name: str,
        group2_name: str,
        save_path: Path,
        title: str,
        use_log: bool = False
    ):
        """
        繪製頭譜圖網格（所有頻帶）
        
        Args:
            group1_powers: 組別1能量字典
            group2_powers: 組別2能量字典
            montage_info: (montage, ch_names) tuple
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            save_path: 儲存路徑
            title: 圖表標題
            use_log: 是否使用對數刻度
        """
        import matplotlib.gridspec as gridspec
        
        montage, ch_names = montage_info
        bands = list(self.BANDS.keys())
        n_bands = len(bands)
        
        # 建立圖表
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(3, n_bands, figure=fig, hspace=0.3, wspace=0.2)
        
        # 為每個頻帶繪製頭譜圖
        for i, band in enumerate(bands):
            # 準備數據
            data1 = group1_powers[band]
            data2 = group2_powers[band]
            
            if use_log:
                data1 = np.log10(data1 + 1)
                data2 = np.log10(data2 + 1)
            
            # 統一色階範圍
            vmin = min(data1.min(), data2.min())
            vmax = max(data1.max(), data2.max())
            
            # 組別1頭譜圖
            ax1 = fig.add_subplot(gs[0, i])
            self._draw_topomap(data1, montage, ch_names, ax1, 
                             f'{band}', vmin, vmax)
            if i == 0:
                ax1.set_ylabel(f'{group1_name}\n(n={len(group1_powers[band])})', 
                             fontsize=12, fontweight='bold')
            
            # 組別2頭譜圖
            ax2 = fig.add_subplot(gs[1, i])
            self._draw_topomap(data2, montage, ch_names, ax2,
                             '', vmin, vmax)
            if i == 0:
                ax2.set_ylabel(f'{group2_name}\n(n={len(group2_powers[band])})', 
                             fontsize=12, fontweight='bold')
            
            # 差異頭譜圖
            ax3 = fig.add_subplot(gs[2, i])
            diff = data2 - data1
            vmax_diff = max(abs(diff.min()), abs(diff.max()))
            self._draw_topomap(diff, montage, ch_names, ax3,
                             '', -vmax_diff, vmax_diff, cmap='RdBu_r')
            if i == 0:
                ax3.set_ylabel(f'Difference\n({group2_name} - {group1_name})', 
                             fontsize=12, fontweight='bold')
        
        fig.suptitle(f'{title} Topographical Maps', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_topomap_difference(
        self,
        group1_abs: Dict[str, np.ndarray],
        group2_abs: Dict[str, np.ndarray],
        group1_rel: Dict[str, np.ndarray],
        group2_rel: Dict[str, np.ndarray],
        montage_info: Tuple,
        group1_name: str,
        group2_name: str,
        save_path: Path
    ):
        """
        繪製差異頭譜圖（專門顯示組間差異）
        
        Args:
            group1_abs: 組別1絕對能量
            group2_abs: 組別2絕對能量
            group1_rel: 組別1相對能量
            group2_rel: 組別2相對能量
            montage_info: (montage, ch_names) tuple
            group1_name: 組別1名稱
            group2_name: 組別2名稱
            save_path: 儲存路徑
        """
        import matplotlib.gridspec as gridspec
        
        montage, ch_names = montage_info
        bands = list(self.BANDS.keys())
        n_bands = len(bands)
        
        # 建立圖表
        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(2, n_bands, figure=fig, hspace=0.3, wspace=0.2)
        
        for i, band in enumerate(bands):
            # 絕對能量差異（使用對數）
            abs_diff = np.log10(group2_abs[band] + 1e-10) - np.log10(group1_abs[band] + 1e-10)
            vmax_abs = max(abs(abs_diff.min()), abs(abs_diff.max()))
            
            ax1 = fig.add_subplot(gs[0, i])
            self._draw_topomap(abs_diff, montage, ch_names, ax1,
                             f'{band}', -vmax_abs, vmax_abs, cmap='RdBu_r')
            if i == 0:
                ax1.set_ylabel('Log Absolute\nPower Difference', 
                             fontsize=11, fontweight='bold')
            
            # 相對能量差異
            rel_diff = group2_rel[band] - group1_rel[band]
            vmax_rel = max(abs(rel_diff.min()), abs(rel_diff.max()))
            
            ax2 = fig.add_subplot(gs[1, i])
            self._draw_topomap(rel_diff, montage, ch_names, ax2,
                             '', -vmax_rel, vmax_rel, cmap='RdBu_r')
            if i == 0:
                ax2.set_ylabel('Relative\nPower Difference', 
                             fontsize=11, fontweight='bold')
        
        fig.suptitle(f'Power Difference: {group2_name} - {group1_name}', 
                    fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _draw_topomap(
        self,
        data: np.ndarray,
        montage,
        ch_names: List[str],
        ax,
        title: str,
        vmin: float,
        vmax: float,
        cmap: str = 'RdBu_r'
    ):
        """
        繪製單個頭譜圖
        
        Args:
            data: 頻道數據
            montage: 電極位置
            ch_names: 頻道名稱
            ax: matplotlib axes
            title: 標題
            vmin: 最小值
            vmax: 最大值
            cmap: 色圖
        """
        # 創建 Info 物件
        info = mne.create_info(
            ch_names=ch_names[:len(data)],
            sfreq=250,
            ch_types='eeg'
        )
        info.set_montage(montage, match_case=False)
        
        # 繪製頭譜圖
        im, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            vlim=(vmin, vmax),
            cmap=cmap,
            sensors=True,
            contours=6,
            outlines='head',
            sphere=None,
            res=128
        )
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # 加入色條（只在最右側）
        if title:  # 只在有標題的圖（第一行）加色條
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    # ========== 工具方法 ==========
    
    @staticmethod
    def inspect_edf_channels(edf_path: Path) -> List[str]:
        """
        檢查 EDF 檔案的頻道名稱（診斷工具）
        
        Args:
            edf_path: EDF 檔案路徑
            
        Returns:
            頻道名稱列表
        """
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        return raw.ch_names