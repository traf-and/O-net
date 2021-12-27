import argparse



parser = argparse.ArgumentParser(description="Train/test ONet interface")

parser.add_argument("-mode", dest="mode", default='test', type=str, help='Train, test or result')
parser.add_argument("-weights", dest="weights_path", default='', type=str, help='Weights pats, default for train')
parser.add_argument("-data", dest="data_path", default='', type=str, help='Data path, if default, donwload 300w and Menpo datasets')
parser.add_argument("-split", dest="split_coef", default=0.9, type=float, help='Train/val coef, in [0;1]')
parser.add_argument("-epochs", dest="epochs", default=10, type=int, help='Num of max epochs')
parser.add_argument("-batch", dest="batch_", default=10, type=int, help='Batch len')

args = parser.parse_args()

print(args)
if args.mode == "train":
    train_mode(args) 
elif args.mode == "test":
    test_mode(args)
elif args.mode == "result"
    result_mode(args)
else:
    print('Unknow mode')
