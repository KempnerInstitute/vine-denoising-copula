"""
Training logger that writes metrics to CSV files for easy monitoring.

Creates detailed logs in logs/ directory:
- training_log.csv: Per-step training metrics
- validation_log.csv: Validation results
- summary.txt: Human-readable summary updated in real-time
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time


class TrainingLogger:
    """
    Logs training metrics to CSV files and console.
    
    Creates:
    - logs/training_log_TIMESTAMP.csv: All training steps
    - logs/validation_log_TIMESTAMP.csv: Validation results
    - logs/training_summary.txt: Latest summary (human-readable)
    """
    
    def __init__(self, log_dir: Path, run_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            prefix = f"{run_name}_{timestamp}"
        else:
            prefix = timestamp
        
        self.train_log_path = self.log_dir / f"training_log_{prefix}.csv"
        self.val_log_path = self.log_dir / f"validation_log_{prefix}.csv"
        self.summary_path = self.log_dir / "training_summary.txt"
        
        # Initialize CSV files
        self.train_csv_initialized = False
        self.val_csv_initialized = False
        
        # Tracking
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.step_times = []
        
        # Create initial summary
        self._write_summary_header()
        
        print("Logging initialized:")
        print(f"  Training log: {self.train_log_path}")
        print(f"  Validation log: {self.val_log_path}")
        print(f"  Summary: {self.summary_path}")
        print()
    
    def log_train_step(self, step: int, metrics: Dict[str, float]):
        """
        Log a single training step.
        
        Args:
            step: Global step number
            metrics: Dictionary of metric name -> value
        """
        # Add timestamp and step time
        current_time = time.time()
        elapsed = current_time - self.start_time
        step_time = current_time - self.last_log_time
        self.last_log_time = current_time
        
        # Track step times for throughput
        self.step_times.append(step_time)
        if len(self.step_times) > 100:
            self.step_times.pop(0)
        
        # Prepare row
        row = {
            'step': step,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_sec': f"{elapsed:.2f}",
            'step_time_ms': f"{step_time * 1000:.2f}",
            **{k: f"{v:.6f}" for k, v in metrics.items()}
        }
        
        # Initialize CSV if needed
        if not self.train_csv_initialized:
            with open(self.train_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            self.train_csv_initialized = True
        
        # Append to CSV
        with open(self.train_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def log_validation(self, step: int, metrics: Dict[str, float]):
        """
        Log validation results.
        
        Args:
            step: Global step number
            metrics: Dictionary of validation metrics
        """
        row = {
            'step': step,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{k: f"{v:.6f}" for k, v in metrics.items()}
        }
        
        # Initialize CSV if needed
        if not self.val_csv_initialized:
            with open(self.val_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            self.val_csv_initialized = True
        
        # Append to CSV
        with open(self.val_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def update_summary(self, step: int, max_steps: int, metrics: Dict[str, float], 
                       val_metrics: Optional[Dict[str, float]] = None):
        """
        Update the human-readable summary file.
        
        Args:
            step: Current step
            max_steps: Total steps
            metrics: Latest training metrics
            val_metrics: Latest validation metrics (optional)
        """
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        # Estimate remaining time
        if step > 0:
            steps_per_sec = step / elapsed
            remaining_steps = max_steps - step
            remaining_sec = remaining_steps / steps_per_sec
            eta_hours = int(remaining_sec // 3600)
            eta_minutes = int((remaining_sec % 3600) // 60)
        else:
            steps_per_sec = 0
            eta_hours = 0
            eta_minutes = 0
        
        # Average step time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        # Format summary
        summary = [
            "=" * 70,
            "TRAINING PROGRESS",
            "=" * 70,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Step: {step:,} / {max_steps:,} ({100 * step / max_steps:.1f}%)",
            f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}",
            f"ETA: {eta_hours:02d}:{eta_minutes:02d}",
            f"Speed: {steps_per_sec:.2f} steps/sec ({avg_step_time * 1000:.1f} ms/step)",
            "",
            "LATEST TRAINING METRICS",
            "-" * 70,
        ]
        
        # Add training metrics
        for key, value in sorted(metrics.items()):
            summary.append(f"  {key:<20}: {value:>12.6f}")
        
        # Add validation metrics if available
        if val_metrics:
            summary.extend([
                "",
                "LATEST VALIDATION METRICS",
                "-" * 70,
            ])
            for key, value in sorted(val_metrics.items()):
                summary.append(f"  {key:<20}: {value:>12.6f}")
        
        summary.extend([
            "",
            "LOG FILES",
            "-" * 70,
            f"  Training: {self.train_log_path.name}",
            f"  Validation: {self.val_log_path.name}",
            "",
            "=" * 70,
        ])
        
        # Write to file
        with open(self.summary_path, 'w') as f:
            f.write('\n'.join(summary))
    
    def _write_summary_header(self):
        """Write initial summary file."""
        with open(self.summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING STARTED\n")
            f.write("=" * 70 + "\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log directory: {self.log_dir}\n")
            f.write("=" * 70 + "\n")
    
    def print_progress(self, step: int, max_steps: int, metrics: Dict[str, float]):
        """
        Print progress to console.
        
        Args:
            step: Current step
            max_steps: Total steps
            metrics: Latest metrics
        """
        # Format metrics string
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Compute progress
        progress_pct = 100 * step / max_steps
        
        # Estimate speed
        elapsed = time.time() - self.start_time
        if step > 0 and elapsed > 0:
            steps_per_sec = step / elapsed
            speed_str = f"{steps_per_sec:.2f} steps/sec"
        else:
            speed_str = "calculating..."
        
        # Print
        print(f"[Step {step:6d}/{max_steps:6d}] ({progress_pct:5.1f}%) | {speed_str} | {metrics_str}")
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        elapsed = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        return {
            'elapsed_sec': elapsed,
            'avg_step_time_ms': avg_step_time * 1000,
            'steps_per_sec': 1.0 / avg_step_time if avg_step_time > 0 else 0,
        }


def read_training_log(log_path: Path) -> List[Dict[str, float]]:
    """
    Read training log CSV file.
    
    Args:
        log_path: Path to training_log.csv
        
    Returns:
        List of dictionaries with metrics per step
    """
    data = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values
            parsed_row = {'step': int(row['step'])}
            for key, value in row.items():
                if key not in ['step', 'timestamp']:
                    try:
                        parsed_row[key] = float(value)
                    except ValueError:
                        parsed_row[key] = value
            data.append(parsed_row)
    return data


def plot_training_curves(log_path: Path, output_path: Optional[Path] = None):
    """
    Plot training curves from log file.
    
    Args:
        log_path: Path to training_log.csv
        output_path: Where to save plot (shows if None)
    """
    import matplotlib.pyplot as plt
    
    data = read_training_log(log_path)
    
    # Extract data
    steps = [row['step'] for row in data]
    
    # Find all metric columns
    metric_keys = [k for k in data[0].keys() if k not in ['step', 'timestamp', 'elapsed_sec', 'step_time_ms']]
    
    # Plot
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric_key in zip(axes, metric_keys):
        values = [row[metric_key] for row in data]
        ax.plot(steps, values, linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_key)
        ax.set_title(f'{metric_key} vs Step')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    # Test the logger
    import tempfile
    import numpy as np
    
    print("Testing TrainingLogger...\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(Path(tmpdir), run_name="test")
        
        # Simulate training
        for step in range(1, 101):
            metrics = {
                'loss': 1.0 / step + 0.1 * np.random.randn(),
                'loss_nll': 0.5 / step + 0.05 * np.random.randn(),
                'loss_ise': 0.3 / step + 0.03 * np.random.randn(),
            }
            
            logger.log_train_step(step, metrics)
            
            if step % 10 == 0:
                logger.print_progress(step, 100, metrics)
                logger.update_summary(step, 100, metrics)
            
            if step % 25 == 0:
                val_metrics = {'val_loss': 0.8 / step, 'val_nll': 0.4 / step}
                logger.log_validation(step, val_metrics)
                logger.update_summary(step, 100, metrics, val_metrics)
            
            time.sleep(0.01)  # Simulate work
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        print(f"\nGenerated files:")
        for f in Path(tmpdir).glob("*"):
            print(f"  - {f.name}")
        
        # Show summary
        print("\nFinal summary:")
        with open(Path(tmpdir) / "training_summary.txt", 'r') as f:
            print(f.read())
