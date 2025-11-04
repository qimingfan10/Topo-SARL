import pickle
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from rl4seg3d.runner import main as runner_main

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="auto_iteration")
def main(cfg):
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='config')

    iterations = cfg.num_iter
    output_path = cfg.output_path
    Path(output_path + "/0/").mkdir(parents=True, exist_ok=True)
    main_overrides = [f"logger.save_dir={output_path}"]
    target_experiment = cfg.target
    source_experiment = cfg.source

    timestamp = datetime.now().timestamp()
    experiment_split_column = f"split_{timestamp}"
    experiment_gt_column = f"Gt_{timestamp}"
    pretrain_path = cfg.get("pretrain_path", None)
    if not pretrain_path:
        # train supervised network for initial actor
        overrides = main_overrides + [f"trainer.max_epochs={cfg.sup_num_epochs}",
                                      f"trainer.devices=2",
                                      f"+trainer.strategy=ddp_find_unused_parameters_true",
                                      f'model.predict_save_dir={None}',  # no predictions here
                                      f"+model.ckpt_path={output_path}/{0}/actor.ckpt",
                                      f"+model.loss.label_smoothing={cfg.sup_loss_label_smoothing}",
                                      f"datamodule={source_experiment}"]
        sub_cfg = compose(config_name=f"supervised_3d_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))

        # prepare dataset with custom split and gt column
        if experiment_split_column != sub_cfg.datamodule.splits_column:
            df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
            # Reset index to make 'study' a regular column if not already done
            if 'study' not in df.columns:
                df = df.reset_index()  # Make 'study' a regular column
                # Rename the index column to 'study' if it was named something else
                if df.columns[0] != 'study':
                    df = df.rename(columns={df.columns[0]: 'study'})
            df[experiment_split_column] = df.loc[:, sub_cfg.datamodule.splits_column]
            # --- 开始修改 ---
            source_gt_col = sub_cfg.datamodule.get('gt_column', None) # 安全地获取 gt_column 的值

            # 检查 source_gt_col 是否有效且存在于 DataFrame 中
            if source_gt_col and source_gt_col in df.columns:
                print(f"Copying GT information from column '{source_gt_col}' to '{experiment_gt_column}'")
                df[experiment_gt_column] = df.loc[:, source_gt_col]
            else:
                # 如果 gt_column 是 None 或在 CSV 中不存在，则创建新列并填充 False
                print(f"Source GT column ('{source_gt_col}') not found or not specified. Creating '{experiment_gt_column}' with default value False.")
                df[experiment_gt_column] = False 
            # --- 结束修改 ---

            df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
            df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
        sub_cfg.datamodule.splits_column = experiment_split_column
        sub_cfg.datamodule.gt_column = experiment_gt_column
        runner_main(sub_cfg)

    # Predict and test (baseline) on target domain
    # Check if supervised actor checkpoint exists; if not, skip pretrain for baseline
    actor0_ckpt = f"{output_path}/{0}/actor.ckpt"
    actor0_exists = Path(actor0_ckpt).exists()

    overrides = main_overrides + cfg.rl_overrides + [f"trainer.max_epochs=0",
                                                     f"trainer.devices=1",
                                                     f"++trainer.strategy=auto",
                                                     f"predict_subset_frac={cfg.rl_num_predict}",
                                                     f"model.actor.actor.pretrain_ckpt={pretrain_path if pretrain_path is not None else (actor0_ckpt if actor0_exists else 'null')}",
                                                     f"model.actor.actor.ref_ckpt={pretrain_path if pretrain_path is not None else (actor0_ckpt if actor0_exists else 'null')}",
                                                     "reward@model.reward=pixelwise_accuracy",  # will not be used
                                                     f"model.actor_save_path={output_path}/{0}/actor.ckpt",  # no need
                                                     f"model.critic_save_path=null",  # no need
                                                     f'model.predict_save_dir={output_path}',
                                                     f"experiment=ppo_{target_experiment}"
                                                     ]
    sub_cfg = compose(config_name=f"RL_3d_runner.yaml", overrides=overrides)
    # prepare dataset with custom split and gt column
    if experiment_split_column != sub_cfg.datamodule.splits_column:
        df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
        # Reset index to make 'study' a regular column if not already done
        if 'study' not in df.columns:
            df = df.reset_index()  # Make 'study' a regular column
            # Rename the index column to 'study' if it was named something else
            if df.columns[0] != 'study':
                df = df.rename(columns={df.columns[0]: 'study'})
        df[experiment_split_column] = df.loc[:, sub_cfg.datamodule.splits_column]
        # 安全地处理 gt 列：如果不存在则创建 False 列
        source_gt_col = sub_cfg.datamodule.get('gt_column', None)
        if source_gt_col and source_gt_col in df.columns:
            df[experiment_gt_column] = df.loc[:, source_gt_col]
        else:
            print(f"Source GT column ('{source_gt_col}') not found or not specified. Creating '{experiment_gt_column}' with default value False.")
            df[experiment_gt_column] = False
        df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
    sub_cfg.datamodule.splits_column = experiment_split_column
    sub_cfg.datamodule.gt_column = experiment_gt_column
    runner_main(sub_cfg)

    for i in range(1, iterations + 1):
        # train reward net
        overrides = main_overrides + [f"trainer.max_epochs={cfg.rn_num_epochs}",
                                      f"trainer.devices=2",
                                      f"+trainer.strategy=ddp_find_unused_parameters_true",
                                      f"datamodule.data_path={output_path}/",
                                      f"model.save_model_path={output_path}/{i - 1}/rewardnet.ckpt",
                                      f"+model.var_file={cfg.var_file}"]
        sub_cfg = compose(config_name=f"reward_3d_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)

        next_output_path = f'{output_path}/{i}/'
        Path(next_output_path).mkdir(parents=True, exist_ok=True)

        # 尝试读取温度等变量；若不存在则使用默认值
        try:
            saved_vars = pickle.load(open(cfg.var_file, "rb"))
        except Exception:
            saved_vars = {"Temperature_factor": 1.0}

        # train PPO model with fresh reward net
        # Resolve previous iteration actor/critic checkpoints if present
        prev_actor_ckpt = f"{output_path}/{i - 1}/actor.ckpt"
        prev_actor_exists = Path(prev_actor_ckpt).exists()
        prev_critic_ckpt = f"{output_path}/{i - 1}/critic.ckpt"
        prev_critic_exists = Path(prev_critic_ckpt).exists()

        overrides = main_overrides + cfg.rl_overrides + \
                    [f"trainer.max_epochs={cfg.rl_num_epochs}",
                     f"trainer.devices=2",
                     f"trainer.strategy=ddp_find_unused_parameters_true",
                     f"predict_subset_frac={cfg.rl_num_predict}",
                     f"datamodule.splits_column={experiment_split_column}",
                     f"datamodule.gt_column={experiment_gt_column}",
                     f"+datamodule.train_batch_size={8 * i}",
                     f"model.actor.actor.pretrain_ckpt={prev_actor_ckpt if prev_actor_exists else 'null'}",
                     f"model.actor.actor.ref_ckpt={actor0_ckpt if actor0_exists else (pretrain_path if pretrain_path is not None else 'null')}",
                     f"model.reward.state_dict_paths.anatomical={output_path}/{i - 1}/rewardnet.ckpt",
                     # f"model.reward.temp_factor={float(saved_vars['Temperature_factor'])}",
                     f"model.actor_save_path={output_path}/{i}/actor.ckpt",
                     f"model.critic_save_path={output_path}/{i}/critic.ckpt",
                     f'model.predict_save_dir={output_path}',
                     f"model.entropy_coeff={max(0.3 / (i * 2), 0)}",
                     f"model.divergence_coeff={0.1 / (i * 2)}",
                     f"experiment=ppo_{target_experiment}"
                     ]
        if prev_critic_exists:
            overrides += [f"model.actor.critic.pretrain_ckpt={prev_critic_ckpt}"]
        sub_cfg = compose(config_name=f"RL_3d_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)


if __name__ == '__main__':
    # Load any available `.env` file
    load_dotenv()
    main()
