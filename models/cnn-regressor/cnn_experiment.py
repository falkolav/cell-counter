from fastai.vision.all import *
import os
import matplotlib.pyplot as plt
import argparse
import datetime

# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=

#Training settings
parser = argparse.ArgumentParser(description='CNN Regressor')

#Arguments
#training
parser.add_argument('--batch_size', type=int, default=32, metavar='BStrain',
                    help='input batch size (default: 32)')
parser.add_argument('--epochs', type=int, default=400, metavar='Epochs',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--image_size', type=int, default=128, metavar='ImgSize',
                    help='size to resize images to (default: 128)')
parser.add_argument('--augmentations', default=True, action='store_true',
                    help='apply augmentations to data (default: True)')

#model
parser.add_argument('--pretrained', default=True, action='store_true',
                    help='load a pretrained xResNet50 model (default: True)')
parser.add_argument('--load_model_from_paper', default=True, action='store_true',
                    help='load the model as trained in the paper (default: True)')
parser.add_argument('--training', default=False, action='store_true',
                    help='run model in training mode (default: False)')
parser.add_argument('--inference', default=True, action='store_true',
                    help='run model in inference mode (default: True)')

#folders
parser.add_argument('--data_folder', type=str, default='/Users/falkolavitt/Python/CNN-regressor/data/', metavar='DF',
                    help='path to folder where the data is located (default: /working_directory/data/)')
parser.add_argument('--models_folder', type=str, default='/Users/falkolavitt/Python/CNN-regressor/models/cnn-regressor/models/', metavar='MF',
                    help='path to folder where the models are located and results are saved (default: models/cnn-regressor/models/)')

#cuda
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='If false, enables CUDA training (default: True')

#seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0} if not args.cuda else {}

# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=
# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]

    model_name = 'model' + '_' + args.model_signature

    #### Set directories for saving
    model_dir = args.models_folder + 'results' + '_' + 'bs-' + str(args.batch_size) + '_' + 'epochs-' + str(args.epochs) + '_' + 'imgsize-' + str(args.image_size) + '_' + 'augmentations-' + str(args.augmentations) + '_pretrained-' + str(args.pretrained) + '/'

    if args.training:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    #### Prepare dataloader
    print('Loading Data')

    # obtain images from folder location
    folder = args.data_folder + 'train/'
    path = Path(folder)
    fnames = get_image_files(path)

    # obtain labels from filenames
    get_y = lambda x: float(str(x).split('.')[-2])
    splitter = RandomSplitter(valid_pct=0.2, seed=42)

    # apply transformations
    item_tfms = [RatioResize(args.image_size)]
    if args.augmentations:
        batch_tfms=[*aug_transforms(mult=0, flip_vert=True, max_rotate=45, min_zoom=0, max_zoom=0, max_warp=0, p_affine=0), Normalize.from_stats(*imagenet_stats)]

    # create datablocks and dataloader
    blocks = (ImageBlock, RegressionBlock)
    block = DataBlock(blocks=blocks,
                      get_items=get_image_files,
                      get_y=get_y,
                      splitter=splitter,
                      item_tfms=item_tfms,
                      batch_tfms=batch_tfms)

    dls = block.dataloaders(path, bs=args.batch_size, num_workers=kwargs['num_workers'])

    # move dataloader to cuda
    if args.cuda:
        dls.cuda()

    #### Model
    print('Loading Model')
    # create xResNet50
    learn = Learner(dls, xresnet50(pretrained=args.pretrained, n_out=1), metrics=mae)

    # move model to cuda
    if args.cuda:
        learn.model = learn.model.cuda()

    # load pretrained model from paper
    if args.load_model_from_paper:
        learn.load(f'{os.getcwd()}/model')

    # train model
    if args.training:
        print('Starting Training')
        learn.fine_tune(args.epochs, cbs=SaveModelCallback(fname=(model_dir + model_name)))

    # load test data into dataloader
    print('Loading test data')
    imgs = get_image_files(f'{args.data_folder}test/')

    if args.cuda:
        dl = learn.dls.test_dl(imgs, with_labels=True)
        dl.cuda()
    else:
        dl = learn.dls.test_dl(imgs, with_labels=True, num_workers=kwargs['num_workers'])

    # print performance on test set
    print('Performance: \nSum of error and Mean Absolute Error:')
    res = learn.validate(dl=dl)
    print(res)

    if args.training:
        # save performance
        f = open(f'{model_dir}{model_name}.txt', 'w+')
        f.write(str(args) + '\n' + str(res))
        f.close()

        # create boxplot of errors
        plt.boxplot(abs(res[0].view((args.batch_size)) - res[1]), labels=['CNN'])
        plt.savefig(f'{model_dir}{model_name}.png')
        plt.close()

if __name__ == "__main__":
    run(args, kwargs)