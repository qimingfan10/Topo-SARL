"""
诊断脚本：检查梯度计算问题
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/RL4Seg3D')

from rl4seg3d.RLmodule_3D import RLmodule3D

# 加载模型
print("加载模型...")
model = RLmodule3D.load_from_checkpoint(
    '/home/ubuntu/RL4Seg3D/lightning_logs/version_237/checkpoints/last.ckpt',
    strict=False
)

print("\n" + "="*80)
print("1. 检查Actor参数的requires_grad状态")
print("="*80)
actor_params_count = 0
actor_trainable_count = 0
for name, param in model.actor.actor.named_parameters():
    actor_params_count += 1
    if param.requires_grad:
        actor_trainable_count += 1
    else:
        print(f"❌ {name}: requires_grad=False")

print(f"\nActor总参数数: {actor_params_count}")
print(f"Actor可训练参数数: {actor_trainable_count}")

print("\n" + "="*80)
print("2. 检查Critic参数的requires_grad状态")
print("="*80)
critic_params_count = 0
critic_trainable_count = 0
for name, param in model.actor.critic.named_parameters():
    critic_params_count += 1
    if param.requires_grad:
        critic_trainable_count += 1
    else:
        print(f"❌ {name}: requires_grad=False")

print(f"\nCritic总参数数: {critic_params_count}")
print(f"Critic可训练参数数: {critic_trainable_count}")

print("\n" + "="*80)
print("3. 检查优化器配置")
print("="*80)
opt_net, opt_critic = model.actor.get_optimizers()

print(f"\nActor优化器类型: {type(opt_net)}")
print(f"Actor优化器参数组数: {len(opt_net.param_groups)}")
for i, group in enumerate(opt_net.param_groups):
    print(f"  组{i}: lr={group['lr']}, eps={group['eps']}, 参数数={len(group['params'])}")
    
print(f"\nCritic优化器类型: {type(opt_critic)}")
print(f"Critic优化器参数组数: {len(opt_critic.param_groups)}")
for i, group in enumerate(opt_critic.param_groups):
    print(f"  组{i}: lr={group['lr']}, eps={group['eps']}, 参数数={len(group['params'])}")

print("\n" + "="*80)
print("4. 检查优化器管理的参数是否正确")
print("="*80)

# 检查actor优化器管理的参数
actor_opt_params = set()
for group in opt_net.param_groups:
    for p in group['params']:
        actor_opt_params.add(id(p))

# 检查实际的actor参数
actor_real_params = set()
for p in model.actor.actor.net.parameters():
    actor_real_params.add(id(p))

print(f"Actor优化器管理的参数数: {len(actor_opt_params)}")
print(f"Actor.net实际参数数: {len(actor_real_params)}")
print(f"参数匹配: {actor_opt_params == actor_real_params}")

print("\n" + "="*80)
print("5. 测试梯度计算")
print("="*80)

# 创建虚拟输入
dummy_input = torch.randn(1, 1, 64, 64, 64).to(model.device)
dummy_actions = torch.randint(0, 3, (1, 64, 64, 64)).to(model.device)

# 前向传播
model.eval()
with torch.enable_grad():
    logits, dist, old_dist = model.actor.actor(dummy_input)
    log_probs = dist.log_prob(dummy_actions)
    loss = -log_probs.mean()
    
    print(f"Loss值: {loss.item()}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    total_grad_norm = 0.0
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.actor.actor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if grad_norm > 0:
                params_with_grad += 1
            else:
                params_without_grad += 1
                print(f"  ⚠️ {name}: grad_norm=0")
        else:
            print(f"  ❌ {name}: grad is None")
            params_without_grad += 1
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"\n总梯度范数: {total_grad_norm}")
    print(f"有梯度的参数数: {params_with_grad}")
    print(f"无梯度的参数数: {params_without_grad}")

print("\n" + "="*80)
print("诊断完成")
print("="*80)

