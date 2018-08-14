import argparse
import os
from train import train_main
from eval import eval_main
from utils import path_create

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    ## gpu
    parser.add_argument("--gpu", type=str, default = "", help="gpu to be used in training")
    ## running model
    parser.add_argument("--type", type=str, default = "train", help="running type, train or eval, default value train")
    ## tffm
    parser.add_argument("--rank", type=int, default = 10, help="rank num for tffm, default value 10")
    parser.add_argument("--save", type="bool", nargs="?", const=True,default=True, help="whether to save model")
    parser.add_argument("--reserve", type="bool", nargs="?", const=True,default=False, help="whether to save data")
    parser.add_argument("--log", type="bool", nargs="?", const=True,default=False, help="whether to save log")
    ## data load
    parser.add_argument("--gap", type=int,default = 30, help="time range for train data, default value 30")
    parser.add_argument("--date", type=str, default="", help = "train data date")
    parser.add_argument("--sample", type="bool", nargs="?", const=True,default=False, help="whether to sample data")
    parser.add_argument("--raw", type="bool", nargs="?", const=True,default=True, help="whether to load raw data")
    ## featurelize
    parser.add_argument("--feature", type=str, default="", help = "feature to be used ")
    ## evaluation
    parser.add_argument("--version", type=str, default="", help = "model version to be used in evaluation")

    args = parser.parse_args()
    return args

def main(args):
    ## set the gpu
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## check the ckpt path
    path_create('ckpt')
    ## check the model path
    path_create('models')
    ## check the data path
    path_create('data')
    ## check the log path
    path_create('log')
    if args.type == 'train':
        train_main(args)
    elif args.type == 'eval':
        eval_main(args)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = add_arguments(parser)
  print(args)
  main(args)





