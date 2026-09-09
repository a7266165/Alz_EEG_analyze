"""
測試 STFT 轉換

使用方式:
    python script/test_stft_transform.py --edf_path <path_to_edf>
"""

import sys
from pathlib import Path
import argparse
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stft_transform import EEGToSTFTImage, STFTConfig
from src.config import WORKSPACE_DIR


def test_single_edf(edf_path: Path, output_dir: Path = None):
    """測試單一 EDF 檔案"""
    
    print("=" * 60)
    print("EEG STFT 轉換測試")
    print("=" * 60)
    print(f"輸入: {edf_path}")
    
    if not edf_path.exists():
        print("✗ 檔案不存在")
        return
    
    if output_dir is None:
        output_dir = WORKSPACE_DIR / "stft_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 轉換
    transformer = EEGToSTFTImage()
    result = transformer.transform_file(edf_path)
    
    # 顯示結果
    print(f"\nSubject: {result['subject_id']}")
    print(f"通道數: {len(result['channels'])}")
    print(f"頻率 bins: {len(result['freqs'])}")
    print(f"時間幀: {len(result['times'])}")
    print(f"頻率範圍: {result['freqs'][0]:.1f} - {result['freqs'][-1]:.1f} Hz")
    print(f"時間長度: {result['times'][-1]:.1f} 秒")
    
    # 檢查單一通道
    first_ch = result['channels'][0]
    img = result['images'][first_ch]
    print(f"\n通道 {first_ch} 圖像:")
    print(f"  形狀: {img.shape}")
    print(f"  數值範圍: [{img.min():.2f}, {img.max():.2f}]")
    print(f"  均值: {img.mean():.4f}")
    print(f"  標準差: {img.std():.4f}")
    
    # 儲存
    np_path = output_dir / f"{result['subject_id']}_stft.npz"
    np.savez_compressed(
        np_path,
        **{f"img_{ch}": result['images'][ch] for ch in result['channels']},
        channels=result['channels'],
        freqs=result['freqs'],
        times=result['times']
    )
    print(f"\n✓ 儲存: {np_path}")
    
    # 視覺化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_channels = len(result['channels'])
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        
        for i, ch in enumerate(result['channels']):
            img = result['images'][ch][0]  # 取第一個通道（三個都一樣）
            
            im = axes[i].imshow(
                img, 
                aspect='auto', 
                cmap='viridis',
                origin='lower',
                extent=[result['times'][0], result['times'][-1],
                       result['freqs'][0], result['freqs'][-1]]
            )
            axes[i].set_title(ch)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Freq (Hz)')
            plt.colorbar(im, ax=axes[i])
        
        # 隱藏多餘的子圖
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"{result['subject_id']} - STFT Log-Power", fontsize=14)
        plt.tight_layout()
        
        img_path = output_dir / f"{result['subject_id']}_stft.png"
        plt.savefig(img_path, dpi=150)
        plt.close()
        print(f"✓ 視覺化: {img_path}")
        
    except Exception as e:
        print(f"⚠ 視覺化失敗: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    test_single_edf(
        Path(args.edf_path),
        Path(args.output_dir) if args.output_dir else None
    )