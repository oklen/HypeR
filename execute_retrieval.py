# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse

from kilt import retrieval
from kilt import kilt_utils as utils


def execute(
    logger, test_config_json, retriever, log_directory, model_name, output_folder
):

    # run evaluation
    retrieval.run(
        test_config_json, retriever, model_name, logger, output_folder=output_folder
    )

import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
def main(args):

    print(f"Load {args.retrieve_config}")
    # load configs
    with open(args.retrieve_config, "r") as fin:
        test_config_json = json.load(fin)
    # create a new directory to log and store results
    log_directory = utils.create_logdir_with_timestamp(args.logdir)
    logger = None

    logger = utils.init_logging(log_directory, args.model_name, logger)
    logger.info("loading {} ...".format(args.model_name))

    if args.model_name == "drqa":
        # DrQA tf-idf
        from kilt.retrievers import DrQA_tfidf

        if args.model_configuration:
            retriever = DrQA_tfidf.DrQA.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = DrQA_tfidf.DrQA.from_default_config(args.model_name)
    elif args.model_name == "dpr":
        # DPR
        from kilt.retrievers import DPR_connector

        if args.model_configuration:
            retriever = DPR_connector.DPR(args.model_name, args)
            # retriever = DPR_connector.DPR.from_config_file(
            #     args.model_name, args.model_configuration
            # )
        else:
            retriever = DPR_connector.DPR.from_default_config(args.model_name)
    elif args.model_name == "dpr_distr":
        # DPR distributed
        from kilt.retrievers import DPR_distr_connector

        if args.model_configuration:
            retriever = DPR_distr_connector.DPR.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            raise "No default configuration for DPR distributed!"
    elif args.model_name == "blink":
        # BLINK
        from kilt.retrievers import BLINK_connector

        if args.model_configuration:
            retriever = BLINK_connector.BLINK.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = BLINK_connector.BLINK.from_default_config(args.model_name)
    elif args.model_name == "bm25":
        # BM25
        from kilt.retrievers import BM25_connector

        retriever = BM25_connector.BM25('bm25', '/home/chotao/models/indexes/lucene-index.msmarco-passage-bm25-b8', 5, 92)
        # if args.model_configuration:
        #     retriever = BM25_connector.BM25.from_config_file(
        #         args.model_name, args.model_configuration
        #     )
        # else:
        #     retriever = BM25_connector.BM25.from_default_config(args.model_name)
    else:
        raise ValueError("unknown retriever model")

    execute(
        logger,
        test_config_json,
        retriever,
        log_directory,
        args.model_name,
        args.output_folder,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_config",
        dest="test_config",
        type=str,
        default="kilt/configs/test_data.json",
        help="Test Configuration.",
    )

    parser.add_argument(
        "--logdir",
        dest="logdir",
        type=str,
        default="logs/ranking/",
        help="logdir",
    )

    # parser.add_argument(
    #     "--model_name",
    #     "-m",
    #     dest="model_name",
    #     type=str,
    #     required=True,
    #     help="retriever model name in {drqa,solr,dpr,blink,bm25}",
    # )

    parser.add_argument(
        "--model_configuration",
        "-c",
        dest="model_configuration",
        type=str,
        default=None,
        help="model configuration",
    )

    # parser.add_argument(
    #     "--output_folder",
    #     "-o",
    #     dest="output_folder",
    #     type=str,
    #     required=True,
    #     help="output folder",
    # )

    # args = parser.parse_args()
    main()
