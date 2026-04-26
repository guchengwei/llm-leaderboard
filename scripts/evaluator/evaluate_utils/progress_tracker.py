"""
評価ハーネスの進行状況とターミナル出力の美化機能
"""
import os
import time
import wandb
from typing import Dict, List, Optional, Any
import json


class EvaluationProgressTracker:
    """評価進行状況の追跡とシンプルなターミナル出力"""
    
    def __init__(self, enabled_benchmarks: List[str]):
        self.enabled_benchmarks = enabled_benchmarks
        self.completed_benchmarks = []
        self.current_benchmark = None
        self.benchmark_results = {}
        self.start_time = time.time()
        
    def start_tracking(self):
        """進行状況の追跡を開始"""
        # ヘッダー表示
        self._show_header()
        
    def start_benchmark(self, benchmark_name: str):
        """ベンチマーク開始"""
        self.current_benchmark = benchmark_name
        self._show_benchmark_start(benchmark_name)
        
    def update_benchmark_progress(self, progress_percent: int):
        """ベンチマーク進行状況を更新"""
        # シンプル実装では何もしない
        pass
            
    def complete_benchmark(self, benchmark_name: str, results: Optional[Dict[str, Any]] = None):
        """ベンチマーク完了"""
        self.completed_benchmarks.append(benchmark_name)
        
        if results:
            self.benchmark_results[benchmark_name] = results
            
        # 結果表示
        self._show_benchmark_completion(benchmark_name, results)
        
    def show_leaderboard_table(self, benchmark_name: str, wandb_run: Optional[Any] = None):
        """W&Bからリーダーボードテーブルを取得して表示"""
        try:
            if wandb_run is None:
                wandb_run = wandb.run
                
            if wandb_run is None:
                print(f"⚠️  W&B run not available for {benchmark_name}")
                return
                
            # W&Bからテーブルを取得
            table_key = f"{benchmark_name}_leaderboard_table"
            
            # W&B APIでテーブルを取得
            api = wandb.Api()
            run = api.run(f"{wandb_run.entity}/{wandb_run.project}/{wandb_run.id}")
            
            # ログからテーブルデータを探す
            for log_entry in run.scan_history(keys=[table_key]):
                if table_key in log_entry:
                    table_data = log_entry[table_key]
                    self._render_leaderboard_table(benchmark_name, table_data)
                    break
            else:
                print(f"⚠️  No leaderboard table found for {benchmark_name}")
                
        except Exception as e:
            print(f"❌ Error displaying leaderboard for {benchmark_name}: {e}")
            
    def _render_leaderboard_table(self, benchmark_name: str, table_data: Any):
        """リーダーボードテーブルをシンプル形式で表示"""
        try:
            if hasattr(table_data, 'data'):
                # W&B Tableオブジェクトの場合
                columns = table_data.columns
                data = table_data.data
            elif isinstance(table_data, dict):
                # 辞書形式の場合
                columns = table_data.get('columns', [])
                data = table_data.get('data', [])
            else:
                print(f"⚠️  Unknown table format for {benchmark_name}")
                return
                
            print(f"\n{'='*80}")
            print(f"🏆 {benchmark_name.upper()} LEADERBOARD")
            print(f"{'='*80}")
            
            # ヘッダーを表示
            header = " | ".join([f"{col:>12}" for col in columns])
            print(header)
            print("-" * len(header))
            
            # データを表示（上位10件のみ）
            for i, row in enumerate(data[:10]):
                row_str = " | ".join([f"{str(cell):>12}" for cell in row])
                rank_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"
                print(f"{rank_indicator} {row_str}")
                
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error rendering table for {benchmark_name}: {e}")
            
    def _show_header(self):
        """ヘッダー表示"""
        print("\n" + "="*80)
        print("🚀 LLM EVALUATION HARNESS")
        print("="*80)
        print(f"📊 Total Benchmarks: {len(self.enabled_benchmarks)}")
        print(f"⏰ Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        enabled_list = ", ".join(self.enabled_benchmarks)
        print(f"🎯 Enabled: {enabled_list}")
        print("="*80 + "\n")
        
    def _show_benchmark_start(self, benchmark_name: str):
        """ベンチマーク開始の表示"""
        emoji = self._get_benchmark_emoji(benchmark_name)
        progress = f"{len(self.completed_benchmarks)+1}/{len(self.enabled_benchmarks)}"
        
        print(f"\n{'='*60}")
        print(f"🚀 [{progress}] STARTING {emoji} {benchmark_name.upper()}")
        print(f"{'='*60}")
        
    def _show_benchmark_completion(self, benchmark_name: str, results: Optional[Dict[str, Any]]):
        """ベンチマーク完了の表示"""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"✅ {benchmark_name.upper()} COMPLETED")
        print(f"{'='*60}")
        print(f"⏱️  Elapsed: {elapsed:.1f}s")
        
        if results:
            print("📊 Key Metrics:")
            # 主要メトリクスを表示
            for key, value in list(results.items())[:5]:  # 最初の5つのメトリクス
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.3f}")
                    
        remaining = len(self.enabled_benchmarks) - len(self.completed_benchmarks)
        print(f"📈 Progress: {len(self.completed_benchmarks)}/{len(self.enabled_benchmarks)} ({remaining} remaining)")
        print(f"{'='*60}\n")
        
    def finish_tracking(self):
        """追跡終了"""            
        total_elapsed = time.time() - self.start_time
        
        # 最終サマリー
        print("\n" + "="*80)
        print("🎉 EVALUATION COMPLETED!")
        print("="*80)
        print(f"⏱️  Total Time: {total_elapsed:.1f}s")
        print(f"✅ Completed: {len(self.completed_benchmarks)}/{len(self.enabled_benchmarks)}")
        
        if self.benchmark_results:
            print("\n📊 Final Results Summary:")
            for benchmark, results in self.benchmark_results.items():
                if results and isinstance(results, dict):
                    # 最初のメトリクスを表示
                    first_metric = next(iter(results.items()))
                    if isinstance(first_metric[1], (int, float)):
                        print(f"  {benchmark}: {first_metric[0]}={first_metric[1]:.3f}")
                        
        print("="*80)

    def _get_benchmark_emoji(self, benchmark_name: str) -> str:
        """ベンチマーク名からシード固定のランダム絵文字を選択"""
        import hashlib
        
        # 評価関連の絵文字リスト
        emojis = [
            '🎯', '🔧', '🐛', '📊', '🛡️', '✅', '🧠', '👁️', 
            '🎲', '📝', '🌸', '🔢', '💻', '🤔', '📖', '💬',
            '📄', '🌐', '❓', '👀', '📏', '⚡', '🚀', '💡',
            '🔍', '⭐', '🎪', '🎨', '🎵', '🎭', '🎬', '🎮'
        ]
        
        # ベンチマーク名をハッシュ化してシードとして使用
        hash_value = hashlib.md5(benchmark_name.encode()).hexdigest()
        # ハッシュの最初の8文字を16進数として解釈し、絵文字インデックスを決定
        seed = int(hash_value[:8], 16)
        emoji_index = seed % len(emojis)
        
        return emojis[emoji_index]


# グローバルトラッカーインスタンス
_global_tracker: Optional[EvaluationProgressTracker] = None


def get_progress_tracker() -> Optional[EvaluationProgressTracker]:
    """グローバルプログレストラッカーを取得"""
    return _global_tracker


def initialize_progress_tracker(enabled_benchmarks: List[str]) -> EvaluationProgressTracker:
    """プログレストラッカーを初期化"""
    global _global_tracker
    _global_tracker = EvaluationProgressTracker(enabled_benchmarks)
    return _global_tracker


def start_benchmark_tracking(benchmark_name: str):
    """ベンチマーク追跡開始"""
    if _global_tracker:
        _global_tracker.start_benchmark(benchmark_name)


def complete_benchmark_tracking(benchmark_name: str, results: Optional[Dict[str, Any]] = None):
    """ベンチマーク追跡完了"""
    if _global_tracker:
        _global_tracker.complete_benchmark(benchmark_name, results)
        # W&B Public API への追加 GraphQL アクセスは本体評価に不要で、
        # ネットワーク不安定時に不要な warning を増やすため既定では無効化する。
        if os.environ.get("NEJUMI_SHOW_WANDB_LEADERBOARD", "").lower() in {"1", "true", "yes"}:
            _global_tracker.show_leaderboard_table(benchmark_name)


def update_benchmark_progress(progress_percent: int):
    """ベンチマーク進行状況更新"""
    if _global_tracker:
        _global_tracker.update_benchmark_progress(progress_percent)


def finish_progress_tracking():
    """プログレス追跡終了"""
    if _global_tracker:
        _global_tracker.finish_tracking()
