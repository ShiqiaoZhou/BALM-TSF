import argparse
import os
import random
import time

import numpy as np
import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models import BALM
from utils.logger import get_logger
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    load_content,
    vali,
)

wandb.login()

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


parser = argparse.ArgumentParser(description="BALM-TSF")

fix_seed = 2021 # 2021 0 42 for 3 runs
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    default="long_term_forecast",
    help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
)
parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
parser.add_argument(
    "--model_id", type=str, required=True, default="test", help="model id"
)
parser.add_argument(
    "--model_comment",
    type=str,
    required=True,
    default="none",
    help="prefix when saving test results",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="BALM",
    help="model name, options: [BALM]",
)
parser.add_argument("--seed", type=int, default=2021, help="random seed")

# data loader
parser.add_argument(
    "--data", type=str, required=True, default="ETTm1", help="dataset type"
)
parser.add_argument(
    "--root_path", type=str, default="./dataset", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; "
    "M:multivariate predict multivariate, S: univariate predict univariate, "
    "MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument("--loader", type=str, default="modal", help="dataset type")
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, "
    "options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], "
    "you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

# forecasting task
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)

# model define
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=16, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=32, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="fixed",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in encoder",
)
parser.add_argument("--patch_len", type=int, default=16, help="patch length")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--prompt_domain", type=int, default=0, help="")
parser.add_argument(
    "--llm_model", type=str, default="GPT2", help="LLM model"
)  # LLAMA, GPT2, BERT deepseek
parser.add_argument(
    "--llm_dim", type=int, default="768", help="LLM model dimension"
)  # LLama7b:4096; GPT2-small:768; BERT-base:768; deepseek:1536
parser.add_argument("--mask_rate", type=float, default=0, help="masking rate")

# optimization
parser.add_argument(
    "--num_workers", type=int, default=10, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument("--align_epochs", type=int, default=10, help="alignment epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=8, help="batch size of model evaluation"
)
parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument(
    "--lradj", type=str, default="type1", help="adjust learning rate"
)  # type1
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument("--llm_layers", type=int, default=6)
parser.add_argument("--percent", type=int, default=100)

# wandb
parser.add_argument("--version_num", default="BALM", type=str)
parser.add_argument("--run_name", default="test", type=str)
parser.add_argument("--wandb_flag", type=int, default=1)
parser.add_argument("--wd_project", default="BALM_test", type=str)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./ds_config_zero2.json")
accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin
)

for ii in range(args.itr):
    # setting record of experiments
    setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        ii,
        args.llm_model,
    )

    run_name = "v{}_r{}_nh{}_dn{}_pl{}_mask{}".format(
        args.version_num,
        args.run_name,
        args.n_heads,
        args.data_path,
        args.pred_len,
        args.mask_rate,
    )

    # refer to: ICLR25-FSCA code
    ### wandb settings
    wandb_group_name = f"{setting}"
    wandb_run_name = f"{run_name}_seed{args.seed}_it{ii}"

    if args.wandb_flag and accelerator.is_local_main_process:
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"{args.wd_project}",
            # We pass a run name (otherwise it'll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_run_name,
            group=wandb_group_name,
            config=args,
        )
    else:
        run = wandb.init(mode="disabled")

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")

    model = BALM.Model(args).float()

    path = os.path.join(
        args.checkpoints, setting + "-" + args.model_comment
    )  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path, exist_ok=True)
    logger = get_logger(path, __name__, "record_s" + str(args.seed) + ".log")
    logger.info(args)
    args.logger = logger

    time_now = time.time()
    trainning_time = 0

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=20, eta_min=1e-8
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = (
        accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )
    )

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        # mode = 'train'
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader)
        ):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = (
                torch.zeros_like(batch_y[:, -args.pred_len :, :])
                .float()
                .to(accelerator.device)
            )
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, i
                        )[0]
                    else:
                        outputs, alignment_loss = model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, i
                        )

                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -args.pred_len :, f_dim:].to(
                        accelerator.device
                    )
                    loss = criterion(outputs, batch_y)
                    loss = loss + alignment_loss
                    if accelerator.is_local_main_process:
                        wandb.log({"Train Loss iter": loss.item()})
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, i)[0]
                else:
                    outputs, alignment_loss = model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                        epoch,
                        i,
                        mode="train",
                        batch_y=batch_y,
                    )

                f_dim = -1 if args.features == "MS" else 0
                if args.mask_rate == 0:
                    batch_y = batch_y[:, -args.pred_len :, f_dim:]
                else:
                    batch_y = batch_y[:, -args.pred_len :, f_dim:]
                    batch_y = torch.cat((batch_x, batch_y), dim=1)
                    batch_y = batch_y[..., f_dim:]
                loss = criterion(outputs, batch_y)
                loss = loss + alignment_loss
                if accelerator.is_local_main_process:
                    wandb.log({"Train Loss iter": loss.item()})
                    if isinstance(alignment_loss, int) and alignment_loss == 0:
                        wandb.log({"Train Alignment Loss iter": alignment_loss})
                    else:
                        wandb.log({"Train Alignment Loss iter": alignment_loss.item()})

                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    )
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time)
                )
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == "TST":
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=False
                )
                scheduler.step()

        # Record training time
        one_epoch_time = time.time() - epoch_time
        trainning_time = trainning_time + one_epoch_time
        accelerator.print(
            "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
        )

        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss, vali_mape_loss, preds_val, trues_val = vali(
            args,
            accelerator,
            model,
            vali_data,
            vali_loader,
            criterion,
            mae_metric,
            epoch,
            mode="vali",
        )
        test_loss, test_mae_loss, test_mape_loss, preds_test, trues_test = vali(
            args,
            accelerator,
            model,
            test_data,
            test_loader,
            criterion,
            mae_metric,
            epoch,
            mode="test",
        )
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss
            )
        )
        if accelerator.is_local_main_process:
            wandb.log(
                {"Train Loss": train_loss, "Vali Loss": vali_loss, "epoch": epoch}
            )
            wandb.log(
                {
                    "test mse": test_loss,
                    "test mae": test_mae_loss,
                    "test_mape": test_mape_loss,
                    "epoch": epoch,
                }
            )

        early_stopping(vali_loss, model, path)
        args.logger.info(
            "Setting: {}, Epoch: {}, Test_MSE: {:.6f}, Test_MAE: {:.6f}, training time: {}".format(
                args.data + "_" + str(args.seq_len) + "_" + str(args.pred_len),
                epoch + 1,
                test_loss,
                test_mae_loss,
                trainning_time,
            )
        )
        f = open(os.path.join(path, "result_s" + str(fix_seed) + ".txt"), "a")
        f.write(args.data + "_" + str(args.seq_len) + "_" + str(args.pred_len) + "\n")
        f.write("MSE: {}, MAE: {}".format(test_loss, test_mae_loss))
        f.write("\n")
        f.close()
        # save_test_results(args, preds_test, trues_test)

        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            # save_test_results(args, preds_test, trues_test)
            break

        if args.lradj != "TST":
            if args.lradj == "COS":
                scheduler.step()
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                )
                if accelerator.is_local_main_process:
                    wandb.log({"lr": model_optim.param_groups[0]["lr"], "epoch": epoch})
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]["lr"]
                    accelerator.print(
                        "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                    )
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=True
                )
                if accelerator.is_local_main_process:
                    wandb.log({"lr": model_optim.param_groups[0]["lr"], "epoch": epoch})

        else:
            accelerator.print(
                "Updating learning rate to {}".format(scheduler.get_last_lr()[0])
            )
    if accelerator.is_local_main_process:
        run.finish()


accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     if os.path.exists(path):
#         del_files(path)  # delete checkpoint files
#         accelerator.print('success delete checkpoints')
#     else:
#         os.makedirs(path, exist_ok=True)
#         accelerator.print('created checkpoints directory')    # del_files(path)  # delete checkpoint files
# accelerator.print('success delete checkpoints')
