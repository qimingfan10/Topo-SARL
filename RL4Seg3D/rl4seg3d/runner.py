import torch
torch.set_float32_matmul_precision('medium')

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything

from patchless_nnunet.utils import log_hyperparameters
from rl4seg3d.utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="RL_3d_runner")
def main(cfg):
    # Load any available `.env` file
    load_dotenv()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    try:
        logger = instantiate(cfg.logger)
    except Exception as e:
        print(f"Falling back to TensorBoardLogger. Could not instantiate configured logger: {e}")
        try:
            save_dir = cfg.logger.save_dir if hasattr(cfg.logger, 'save_dir') else str(Path.cwd() / 'logs')
            name = cfg.logger_run_name if hasattr(cfg, 'logger_run_name') else 'run'
        except Exception:
            save_dir, name = str(Path.cwd() / 'logs'), 'run'
        logger = TensorBoardLogger(save_dir=save_dir, name=name)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if isinstance(trainer.logger, CometLogger):
        logger.experiment.log_asset_folder(".hydra", log_file_name=True)
        if cfg.get("comet_tags", None):
            logger.experiment.add_tags(list(cfg.comet_tags))

    if logger:
        print("Logging hyperparams")
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
        log_hyperparameters(object_dict)

    if cfg.train:
        trainer.fit(train_dataloaders=datamodule, model=model)

    # Select checkpoint path for test/predict with safe fallbacks
    if cfg.trainer.max_epochs > 0 and cfg.train:
        ckpt_path = None
        try:
            ckpt_cbs = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
            if ckpt_cbs:
                cb = ckpt_cbs[0]
                # Prefer concrete file paths if available
                if getattr(cb, "best_model_path", None) and Path(cb.best_model_path).exists():
                    ckpt_path = cb.best_model_path
                    print(f"Using best checkpoint file: {ckpt_path}")
                elif getattr(cb, "last_model_path", None) and Path(cb.last_model_path).exists():
                    ckpt_path = cb.last_model_path
                    print(f"Using last checkpoint file: {ckpt_path}")
                else:
                    # Fall back to Lightning aliases if files not known yet
                    if getattr(cb, "monitor", None):
                        ckpt_path = "best"
                        print("Using ckpt_path='best' (monitor set). If unavailable, Lightning will raise; we will retry.")
                    elif getattr(cb, "save_last", False):
                        ckpt_path = "last"
                        print("Using ckpt_path='last' (save_last enabled).")
        except Exception as e:
            print(f"Could not infer checkpoint path from ModelCheckpoint: {e}")
            ckpt_path = None
    elif getattr(cfg, "test_from_ckpt", None):
        ckpt_path = cfg.test_from_ckpt
    else:
        ckpt_path = None

    # test with everything
    datamodule.hparams.subset_frac = 1.0
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, inference_mode=False)
    try:
        trainer.test(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)
    except Exception as e:
        msg = str(e)
        if 'ckpt_path="best"' in msg or 'ckpt_path="last"' in msg:
            print(f"Warning: {e}. Retrying test with ckpt_path=None (current weights).")
            try:
                trainer.test(model=model, dataloaders=datamodule, ckpt_path=None)
            except Exception as e2:
                print(f"Warning: Test failed even with current weights: {e2}. Skipping test phase.")
        elif 'No `test_step()` method defined' in msg:
            print("Warning: Model has no test_step(); skipping test phase.")
        else:
            print(f"Warning: Test failed with error: {e}. Skipping test phase.")

    if getattr(cfg.model, "predict_save_dir", None) and cfg.predict_subset_frac > 0:
        # 先构建predict集长度，若小于设备数则改用单卡+auto策略预测，避免DDP采样断言
        datamodule.hparams.subset_frac = cfg.predict_subset_frac
        try:
            datamodule.setup("predict")
        except Exception:
            pass
        try:
            pred_len = len(getattr(datamodule, 'data_pred', []))
        except Exception:
            pred_len = 0

        world_size = getattr(trainer, 'world_size', 1)
        num_devices = getattr(trainer, 'num_devices', world_size)
        needs_single_device = (pred_len > 0) and (num_devices and pred_len < int(num_devices))

        if needs_single_device:
            print(f"Predict dataset too small for DDP (len={pred_len} < devices={num_devices}). Falling back to single-device predict.")
            single_device_trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, devices=1, strategy="auto")
            try:
                single_device_trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)
            except Exception as e:
                msg = str(e)
                if 'ckpt_path="best"' in msg or 'ckpt_path="last"' in msg:
                    print(f"Warning: {e}. Retrying predict with ckpt_path=None (current weights).")
                    single_device_trainer.predict(model=model, dataloaders=datamodule, ckpt_path=None)
                elif 'No `predict_step()` method defined' in msg or 'requires `forward` method to run' in msg:
                    print("Warning: Model has no predict_step(); skipping predict phase.")
                else:
                    raise
        else:
            try:
                trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)
            except Exception as e:
                msg = str(e)
                if 'ckpt_path="best"' in msg or 'ckpt_path="last"' in msg:
                    print(f"Warning: {e}. Retrying predict with ckpt_path=None (current weights).")
                    trainer.predict(model=model, dataloaders=datamodule, ckpt_path=None)
                elif 'No `predict_step()` method defined' in msg or 'requires `forward` method to run' in msg:
                    print("Warning: Model has no predict_step(); skipping predict phase.")
                else:
                    # 再次兜底：若触发分布式采样断言，也回退到单卡
                    if 'num_samples' in msg or 'UnrepeatedDistributedSamplerWrapper' in msg or 'AssertionError' in msg:
                        print(f"Warning: {e}. Falling back to single-device predict.")
                        single_device_trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, devices=1, strategy="auto")
                        single_device_trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)
                    else:
                        raise
        # 手动保存全部预测三件套作为兜底（某些情况下 pl 不会调用自定义 predict_step）
        try:
            if getattr(cfg.model, "predict_save_dir", None):
                # 仅在单进程或主进程执行，避免重复
                if getattr(trainer, 'world_size', 1) == 1 or getattr(trainer, 'global_rank', 0) == 0:
                    datamodule.setup("predict")
                    from rl4seg3d.utils.file_utils import save_to_reward_dataset
                    from pathlib import Path as _P
                    save_root = (str(cfg.model.predict_save_dir).rstrip('/') + '/rewardDS')
                    _P(save_root).mkdir(parents=True, exist_ok=True)
                    device = next(model.actor.actor.net.parameters()).device
                    with torch.no_grad():
                        for b_idx, batch in enumerate(datamodule.predict_dataloader()):
                            imgs = batch.get('img')
                            if imgs is None:
                                continue
                            imgs = imgs.to(device)
                            gt = batch.get('gt', None)
                            if gt is None:
                                gt = batch.get('approx_gt', None)
                            meta = batch.get('image_meta_dict', {}) or {}
                            # 展开窗口维度到批次，确保 actor 输入为 [N, C, H, W, D]
                            imgs_to_act = imgs
                            windows_count = 1
                            if imgs.dim() == 6:
                                # [B, W, C, H, W2, D] -> [B*W, C, H, W2, D]
                                bsz, nwin, cch, hh, ww, dd = imgs.shape
                                imgs_to_act = imgs.view(bsz * nwin, cch, hh, ww, dd)
                                windows_count = bsz * nwin
                            elif imgs.dim() == 5:
                                windows_count = imgs.shape[0]
                            else:
                                raise ValueError(f"Unexpected imgs ndim={imgs.dim()} shape={tuple(imgs.shape)}")

                            actions = model.actor.act(imgs_to_act, sample=False)

                            # 保存第一个窗口
                            import numpy as np
                            def _to3d(x):
                                x = x.detach().cpu().numpy()
                                x = np.squeeze(x)
                                if x.ndim == 4 and x.shape[0] == 1:
                                    x = x[0]
                                if x.ndim != 3:
                                    raise ValueError(f"Expect 3D volume, got shape={x.shape}")
                                return x

                            # 遍历窗口批次维度逐一保存
                            num_windows = int(actions.shape[0]) if hasattr(actions, 'shape') else windows_count
                            # 获取样本ID
                            case_id = f'case_{b_idx}'
                            if isinstance(meta, dict):
                                case_id = meta.get('case_identifier', case_id)
                            elif isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
                                case_id = meta[0].get('case_identifier', case_id)
                            for w in range(num_windows):
                                try:
                                    img_np = _to3d(imgs_to_act[w, 0])
                                    pred_np = _to3d(actions[w])
                                    if gt is not None:
                                        # 对齐 gt 到 [N, H, W, D]
                                        if gt.dim() == 6:
                                            # [B, W, H, W2, D]
                                            bsz2, nwin2, gh, gw, gd = gt.shape
                                            gt_flat = gt.view(bsz2 * nwin2, gh, gw, gd)
                                            gt_np = _to3d(gt_flat[w])
                                        elif gt.dim() == 5:
                                            # [N, 1, H, W, D] 或 [N, H, W, D]
                                            if gt.shape[1] == 1:
                                                gt_np = _to3d(gt[w, 0])
                                            else:
                                                gt_np = _to3d(gt[w])
                                        elif gt.dim() == 4:
                                            gt_np = _to3d(gt)
                                        else:
                                            gt_np = np.zeros_like(pred_np, dtype=pred_np.dtype)
                                    else:
                                        gt_np = np.zeros_like(pred_np, dtype=pred_np.dtype)
                                    fname = f"{case_id}_w{w:03d}.nii.gz"
                                    save_to_reward_dataset(save_root, fname, img_np.astype(np.float32), gt_np.astype(np.int16), pred_np.astype(np.int16))
                                    print(f"[runner] Fallback saved predict triplet -> {save_root}/(images|gt|pred)/{fname}")
                                except Exception as se:
                                    print(f"[runner] Fallback save failed for {case_id} window {w}: {se}")
        except Exception:
            pass

        if cfg.get("save_csv_after_predict", None) and trainer.world_size > 1 and trainer.global_rank == 0:
            for p in Path(f"{model.temp_files_path}/").glob("temp_pred_*.csv"):
                df = pd.read_csv(p, index_col=0)
                datamodule.df.loc[df.index] = df
                os.remove(p)
            datamodule.df.to_csv(cfg.save_csv_after_predict)


if __name__ == "__main__":
    load_dotenv()
    main()
