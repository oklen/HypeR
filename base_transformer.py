# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import os
import random
import datetime

import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import torch
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
from transformers.generation_utils import GenerationMixin
from data import remove_preds
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor

def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class BaseTransformer(pl.LightningModule, GenerationMixin):
    def __init__(self, hparams: argparse.Namespace, num_labels=None, **config_kwargs):
        "Initialize a model."

        super().__init__()
        self.hparams.update(dict(hparams))
        self.epoch_count = 0

        # cache_dir = hparams.cache_dir if hparams.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            hparams.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            # cache_dir=cache_dir,
            **config_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.model_name_or_path,
            max_seq_length= hparams.max_sequence_length,
            # cache_dir=cache_dir,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            hparams.model_name_or_path,
            config=self.config
            # cache_dir=cache_dir,
        )

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        # model = self.question_encoder
        # if self.hparams.use_prompt:
        #     self.hparams.weight_decay = 0

        model = self
        no_decay = ["bias", "LayerNorm.weight"]
        # attention_comp = ["P_Adapter"]
        prompt_comp = ["prompt_embeddings","prefix_encoder","query_adapter"]
        optimizer_grouped_parameters = [
            dict(
                params=[
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in prompt_comp)
                ],
                weight_decay=self.hparams.weight_decay,
            ),
            {
                "params": [
                    p
                    for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in prompt_comp)
                    
                ],
                "weight_decay": 0.0,
            },
            # {
            #     "params": [
            #         p
            #         for n, p in model.named_parameters()
            #         if any(nd in n for nd in attention_comp)
            #     ],
            #     "learing_rate": self.hparams.module_learning_rate,
            # }
        ]
        optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in prompt_comp) and any(nd in n for nd in no_decay)
            ],
            "learing_rate": self.hparams.module_learning_rate, "weight_decay": 0.0,
        })
        optimizer_grouped_parameters.append(
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in prompt_comp) and  not any(nd in n for nd in no_decay)
            ],
            "learing_rate": self.hparams.module_learning_rate, "weight_decay": 0.0,
        }
        )

        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None,on_tpu=None,using_native_amp=None,using_lbfgs=None
    ):
        # for name,param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        if on_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        avg_loss = getattr(self.trainer, "avg_loss", 0.0)
        tqdm_dict = {
            "loss": "{:.3f}".format(avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        return tqdm_dict

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    # def train_dataloader(self):
    #     train_batch_size = self.hparams.train_batch_size
    #     dataloader = self.load_dataset("train", train_batch_size)

    #     t_total = (
    #         (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.n_gpu)))
    #         // self.hparams.gradient_accumulation_steps
    #         * float(self.hparams.num_train_epochs)
    #     )
    #     scheduler = get_linear_schedule_with_warmup(
    #         self.opt,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=t_total,
    #     )
    #     self.lr_scheduler = scheduler
    #     return dataloader

    # def val_dataloader(self):
    #     return self.load_dataset("dev", self.hparams.eval_batch_size)

    # def test_dataloader(self):
    #     return self.load_dataset("test", self.hparams.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=3,
            type=int,
            help="Total number of training epochs to perform.",
        )


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
        if pl_module.hparams.dense_model and pl_module.hparams.dis_faiss:
            pl_module.retriever.client_index.client.async_mode(False)
        pl_module.epoch_count += 1

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # remove_preds(pl_module.hparams.output_dir, pl_module.hparams.dataset, 'dev')
        if pl_module.hparams.dense_model and pl_module.hparams.dis_faiss:
                pl_module.retriever.client_index.client.async_mode(True)
        
        
    # def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     logger.info("***** Test results *****")

    #     if pl_module.is_logger():
    #         metrics = trainer.callback_metrics

    #         # Log and save results to file
    #         output_test_results_file = os.path.join(
    #             pl_module.hparams.output_dir, "test_results.txt"
    #         )
    #         with open(output_test_results_file, "w") as writer:
    #             for key in sorted(metrics):
    #                 if key not in ["log", "progress_bar"]:
    #                     logger.info("{} = {}\n".format(key, str(metrics[key])))
    #                     writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )


def generic_train(model: BaseTransformer, args: argparse.Namespace):
    # init model
    set_seed(args)

    # if (
    #     os.path.exists(args.output_dir)
    #     and os.listdir(args.output_dir)
    #     and args.do_train
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(
    #             args.output_dir
    #         )
    #     )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
    )
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = []
    callbacks.append(LoggingCallback())
    # callbacks.append(lr_monitor)
    # Append model checkpoint if necessary
    # callbacks.append(checkpoint_callback())
    
    if args.selected_gpu:
        gpus = [6]
        args.n_gpu = gpus
    else:
        gpus = args.n_gpu

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=gpus,
        devices=gpus,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        # checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    # train_params["strategy"] = "ddp_find_unused_parameters_false" # Disable unused parameters checking
    # train_params["strategy"] = "ddp_sharded_find_unused_parameters_false" # Disable unused parameters checking
    ddp = DDPStrategy(find_unused_parameters=False)
    train_params["strategy"] = ddp
    train_params["num_sanity_val_steps"] = args.num_sanity_val_steps
    train_params["log_every_n_steps"] = 5
    train_params["accelerator"] = "gpu"
    train_params["check_val_every_n_epoch"] = args.check_val_every_n_epoch
    # train_params["reload_dataloaders_every_n_epochs"] = args.reload_dataloaders_every_n_epochs
    # train_params["plugins"] = DDPPlugin()

    # train_params["val_check_interval"] = 2

    if args.profile_mode:
        train_params["profiler"] = "simple"
        train_params["overfit_batches"] = 10
    else:
        pass
        # train_params["callbacks"].append(EarlyStopping(monitor="val_loss", mode="min"))

    # train_params["strategy"] = "ddp_spawn_find_unused_parameters_false" # Disable unused parameters checking
    # if args.n_gpu > 1:
    #     train_params["distributed_backend"] = "ddp"
    
    trainer = pl.Trainer(**train_params)
    model.trainer = trainer
    model.barrier = ddp.barrier
    if args.do_train:
        trainer.fit(model)
    else:
        trainer.validate(model)

    return trainer
