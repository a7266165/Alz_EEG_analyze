"""
EEG STFT 圖像轉換模組

將多通道 EEG 訊號轉換為 2D 圖像，供 Vision Foundation Model 使用
策略：每個通道獨立處理，產生獨立的頻譜圖像

參考 Vision4PPG 論文的轉換方法
"""

import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Literal
from dataclasses import dataclass
from scipy.signal import stft


@dataclass
class STFTConfig:
    """STFT 轉換配置"""
    # STFT 參數
    nperseg: int = 256
    noverlap: Optional[int] = None  # None 時為 nperseg // 2
    nfft: int = 256
    
    # 頻率範圍 (Hz)
    freq_min: float = 1.0
    freq_max: float = 50.0


# 預設頻道
DEFAULT_CHANNELS = [
    'F3', 'F4', 'F7', 'F8', 'Fz',
    'C3', 'C4', 'Cz',
    'P3', 'P4', 'Pz',
    'T3', 'T4', 'T5', 'T6',
    'O1', 'O2'
]


class EEGToSTFTImage:
    """
    將 EEG 訊號轉換為 STFT 頻譜圖像
    
    每個通道獨立處理，產生獨立的 (3, n_freq, n_time) 圖像
    """
    
    def __init__(
        self,
        channels: List[str] = None,
        resample_freq: float = 250.0,
        config: STFTConfig = None
    ):
        self.channels = channels or DEFAULT_CHANNELS
        self.resample_freq = resample_freq
        self.config = config or STFTConfig()
    
    def transform_file(self, edf_path: Path) -> Dict:
        """
        轉換單一 EDF 檔案
        
        Returns:
            {
                'subject_id': str,
                'images': {channel_name: (3, n_freq, n_time) ndarray},
                'channels': List[str],
                'freqs': ndarray,
                'times': ndarray,
                'sfreq': float
            }
        """
        # 1. 載入 EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        matched_channels = self._match_channels(raw.ch_names)
        
        if not matched_channels:
            raise ValueError(f"找不到匹配的頻道: {raw.ch_names}")
        
        raw.pick(matched_channels)
        
        if raw.info['sfreq'] != self.resample_freq:
            raw.resample(self.resample_freq)
        
        data = raw.get_data()  # (n_channels, n_samples)
        
        # 2. 對每個通道獨立處理
        images = {}
        freqs_out = None
        times_out = None
        
        for i, ch_name in enumerate(matched_channels):
            signal = data[i]  # (n_samples,)
            
            # STFT
            freqs, times, Zxx = stft(
                signal,
                fs=self.resample_freq,
                nperseg=self.config.nperseg,
                noverlap=self.config.noverlap or self.config.nperseg // 2,
                nfft=self.config.nfft
            )
            
            # 頻率篩選
            freq_mask = (freqs >= self.config.freq_min) & (freqs <= self.config.freq_max)
            Zxx_filtered = Zxx[freq_mask, :]
            
            if freqs_out is None:
                freqs_out = freqs[freq_mask]
                times_out = times
            
            # Log-power
            power = np.abs(Zxx_filtered) ** 2
            log_power = np.log(power + 1e-10)
            
            # Z-score
            mean = np.mean(log_power)
            std = np.std(log_power)
            if std > 1e-10:
                normalized = (log_power - mean) / std
            else:
                normalized = log_power - mean
            
            # 複製三通道
            image = np.stack([normalized, normalized, normalized], axis=0)
            
            # 清理通道名稱
            clean_name = self._clean_channel_name(ch_name)
            images[clean_name] = image.astype(np.float32)
        
        return {
            'subject_id': edf_path.stem,
            'images': images,
            'channels': [self._clean_channel_name(ch) for ch in matched_channels],
            'freqs': freqs_out,
            'times': times_out,
            'sfreq': self.resample_freq
        }
    
    def _match_channels(self, available: List[str]) -> List[str]:
        """匹配頻道名稱"""
        matched = []
        for target in self.channels:
            for pattern in [target, target.upper(), target.lower(),
                          f"EEG {target}-REF", f"EEG {target.upper()}-REF"]:
                if pattern in available:
                    matched.append(pattern)
                    break
        return matched
    
    def _clean_channel_name(self, name: str) -> str:
        """清理頻道名稱"""
        if 'EEG' in name and '-REF' in name:
            return name.replace('EEG ', '').replace('-REF', '')
        return name


if __name__ == "__main__":
    # 簡單測試
    print("測試 EEGToSTFTImage...")
    
    transformer = EEGToSTFTImage()
    
    # 模擬單通道處理
    np.random.seed(42)
    n_samples = 250 * 10  # 10 秒
    t = np.arange(n_samples) / 250.0
    
    # 合成訊號：10Hz alpha + 20Hz beta
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # STFT
    freqs, times, Zxx = stft(signal, fs=250, nperseg=256, nfft=256)
    
    # 頻率篩選
    freq_mask = (freqs >= 1) & (freqs <= 50)
    Zxx_filtered = Zxx[freq_mask, :]
    freqs_filtered = freqs[freq_mask]
    
    # Log-power + z-score
    log_power = np.log(np.abs(Zxx_filtered) ** 2 + 1e-10)
    normalized = (log_power - np.mean(log_power)) / np.std(log_power)
    
    # 複製三通道
    image = np.stack([normalized] * 3, axis=0)
    
    print(f"輸入訊號: {signal.shape}")
    print(f"STFT 原始: {Zxx.shape}")
    print(f"頻率篩選後: {Zxx_filtered.shape}")
    print(f"輸出圖像: {image.shape}")
    print(f"頻率範圍: {freqs_filtered[0]:.1f} - {freqs_filtered[-1]:.1f} Hz")
    print(f"數值範圍: [{image.min():.2f}, {image.max():.2f}]")
    print("\n✓ 測試完成")