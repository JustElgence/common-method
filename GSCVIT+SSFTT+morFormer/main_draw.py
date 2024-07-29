import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset, sample_gt_num, HSIDataset1
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test, test1
from utils.utils import Draw
import time
import seaborn as sns
import visdom


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")
    print(palette)
    # quit()
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_to_color(x, N_Classes, palette):
    
    if palette is None:
    # Generate color palette
        palette = {0: (0, 0, 0)}

    for k, color in enumerate(sns.color_palette("hls", N_Classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    invert_palette = {v: k for k, v in palette.items()}
    
    return convert_to_color_(x, palette=palette)
    
def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:

        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:

        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='gscvit') # model name
    parser.add_argument("--dataset_name", type=str, default="sa") # dataset name
    parser.add_argument("--dataset_dir", type=str, default="./datasets") # dataset dir
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--patch_size", type=int, default=8) # patch_size
    parser.add_argument("--num_run", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs", type=int, default=128)  # bs = batch size
    parser.add_argument("--ratio", type=float, default=0.02) # ratio of training + validation sample

    opts = parser.parse_args()

    device = torch.device("cuda:1")

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training, {} for validation and {} testing".format(opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    viz = visdom.Visdom(env= opts.model + '  ' + opts.dataset_name)
    if not viz.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)

    num_bands = image.shape[-1]

    # random seeds
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410]

    # empty list to storing results
    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run + 1, opts.num_run))

        # get train_gt, val_gt and test_gt
        # when sampling by num, use sample_gt_num
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])
        del trainval_gt

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)

        test_set = HSIDataset1(image, test_gt, patch_size=opts.patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, opts.bs, drop_last=False, shuffle=False)


        # load model and loss
        model = get_model(opts.model, opts.dataset_name, opts.patch_size)

        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)
            print("network information:")
            # with torch.no_grad():
            #     summary(model, torch.zeros((1, 1, num_bands, opts.patch_size, opts.patch_size)))

        model = model.to(device)
        # print(model)
        optimizer, scheduler = load_scheduler(opts.model, model)

        criterion = nn.CrossEntropyLoss()

        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        tic1 = time.time()
        try:
            train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')
        tic2 = time.time()

        # test the model
        toc1 = time.time()
        # probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
        acc, probabilities, matrix = test1(model, model_dir, image, opts.patch_size, num_classes, device, test_loader)
        toc2 = time.time()

        print(acc)
        prediction = np.argmax(probabilities, axis=-1)
        display_predictions(convert_to_color(matrix, num_classes + 1, palette = None), viz, caption="prediction")

        # computing metrics
        run_results = metrics(probabilities, test_gt, n_classes=num_classes)  # only for test set
        run_results["Tr"] = tic2 - tic1
        run_results["Te"] = toc2 - toc1
        results.append(run_results)
        show_results(run_results, label_values=labels)

        # draw the classification map
        # Draw(model,image,gt,opts.patch_size,opts.dataset_name,opts.model,num_classes)

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        show_results(results, label_values=labels, agregated=True)