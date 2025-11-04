#!/usr/bin/env python3
"""查看训练日志和指标"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='查看TensorBoard日志')
    parser.add_argument('--log-dir', required=True, help='日志目录路径')
    args = parser.parse_args()
    
    log_path = Path(args.log_dir)
    
    if not log_path.exists():
        print(f"错误: 日志目录不存在: {log_path}")
        return 1
    
    print("="*60)
    print("训练日志分析")
    print("="*60)
    
    # 查找事件文件
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    
    if not event_files:
        print("未找到TensorBoard事件文件")
        return 1
    
    print(f"\n找到 {len(event_files)} 个事件文件:")
    for ef in event_files[:5]:  # 只显示前5个
        print(f"  - {ef.relative_to(log_path.parent)}")
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # 分析最新的事件文件
        latest_event = max(event_files, key=lambda x: x.stat().st_mtime)
        print(f"\n分析最新事件文件:")
        print(f"  {latest_event.name}")
        
        ea = EventAccumulator(str(latest_event))
        ea.Reload()
        
        # 显示可用的标量
        scalar_tags = ea.Tags().get('scalars', [])
        
        if scalar_tags:
            print(f"\n可用的训练指标 ({len(scalar_tags)}个):")
            for tag in sorted(scalar_tags)[:20]:  # 只显示前20个
                print(f"  - {tag}")
            
            # 显示一些关键指标的最终值
            print("\n关键指标最终值:")
            key_metrics = ['reward', 'advantage', 'loss', 'val_loss', 'test_dice', 'test_acc']
            
            for metric in key_metrics:
                matching_tags = [t for t in scalar_tags if metric.lower() in t.lower()]
                for tag in matching_tags[:3]:  # 每个指标最多显示3个
                    try:
                        events = ea.Scalars(tag)
                        if events:
                            last_value = events[-1].value
                            last_step = events[-1].step
                            print(f"  {tag:30s}: {last_value:.4f} (step {last_step})")
                    except:
                        pass
        else:
            print("\n未找到标量指标")
        
        print("\n" + "="*60)
        print("提示: 完整的训练曲线需要使用TensorBoard查看")
        print("安装: pip install tensorboard")
        print(f"运行: tensorboard --logdir={log_path.parent}")
        print("="*60)
        
    except ImportError:
        print("\n提示: 需要安装tensorboard来查看详细日志")
        print("运行: pip install tensorboard")
        
        # 显示基本信息
        print(f"\n检查点文件:")
        ckpt_files = list(log_path.rglob("*.ckpt"))
        for ckpt in ckpt_files[:5]:
            size_mb = ckpt.stat().st_size / (1024*1024)
            print(f"  - {ckpt.name}: {size_mb:.2f} MB")
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

