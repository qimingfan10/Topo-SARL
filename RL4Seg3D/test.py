import numpy as np

print(f"当前 NumPy 版本: {np.__version__}")

try:
    # 尝试导入 sam-2 包
    # 注意：这里的包名可能是 'sam' 或 'segment_anything' 或其他，
    # 取决于您安装 sam-2 时实际的包名。请根据实际情况修改。
    # 假设包名是 'segment_anything' (这是官方 SAM 的包名)
    from rl4seg3d.PPO_3d import PPO3D 
    print("成功导入 segment_anything (sam-2) 包！")
    
    # (可选) 尝试调用一个 sam 的基本功能，例如加载模型
    # 这需要您知道 sam-2 的基本用法，并可能需要下载模型权重
    # try:
    #     from segment_anything import sam_model_registry, SamPredictor
    #     sam_checkpoint = "path/to/your/sam_vit_h_....pth" # 您需要提供模型路径
    #     model_type = "vit_h"
    #     device = "cuda" # or "cpu"
    #     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    #     sam.to(device=device)
    #     print("成功加载 SAM 模型！")
    # except Exception as e:
    #     print(f"导入成功，但调用 SAM 功能时出错: {e}")
    #     print("这可能仍然是由于 NumPy 版本不兼容或其他依赖问题。")

except ImportError as e:
    print(f"导入 sam-2 包失败: {e}")
    print("这很可能是由于 NumPy 版本不兼容导致的。sam-2 在当前环境下不可用。")
except Exception as e:
    print(f"尝试导入 sam-2 时发生其他错误: {e}")
    print("这也可能与 NumPy 版本或其他依赖冲突有关。sam-2 在当前环境下可能不可用。")
