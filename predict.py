import argparse
import os
import cv2
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from PIL import Image

from unet import *
from utils import hwc_to_chw, split_img_into_squares, merge_masks
from utils import dense_crf
# from utils import plot_img_and_mask

from torchvision import transforms
import glob

def predict_img(net,
                full_img,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False,
                channels=2):

    img_height = full_img.shape[0]
    img_width = full_img.shape[1]

    left_square, right_square = split_img_into_squares(img.copy() / 255)

    # left_square = hwc_to_chw(left_square)
    # right_square = hwc_to_chw(right_square)
    #
    # X_left = torch.from_numpy(left_square).unsqueeze(0)
    # X_right = torch.from_numpy(right_square).unsqueeze(0)
    normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.4, 0.4, 0.4])
    transforms4imgs = T.Compose([
        T.ToTensor(),
        normalize
    ])
    X_left = transforms4imgs(left_square)
    X_right = transforms4imgs(right_square)


    X_left = Variable(X_left).unsqueeze(0)
    X_right = Variable(X_right).unsqueeze(0)

    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    net.eval()

    output_left = net(X_left)
    output_right = net(X_right)

    if channels > 1:
        left_probs = torch.argmax(output_left, dim=1).float()
        right_probs = torch.argmax(output_right, dim=1).float()
    else:
        left_probs = F.sigmoid(output_left).squeeze(0)
        right_probs = F.sigmoid(output_right).squeeze(0)

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_height),
            transforms.ToTensor()
        ]
    )

    left_probs = tf(left_probs.cpu())
    right_probs = tf(right_probs.cpu())

    left_mask_np = left_probs.squeeze().cpu().numpy()
    right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='checkpoints/CP43.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
    #                     help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(imgs_path):
    in_files = imgs_path
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    import os;os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_args()
    filapath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      'data/test')

    in_files = glob.glob(filapath + "/*." + 'bmp')
    out_files = get_output_filenames(in_files)

    out_channels = 2

    net = DeeperUNet(n_channels=3, n_classes=out_channels, SE_mode=True)

    print("Loading model {}".format(args.model))
    # original saved file with DataParallel
    if args.cpu:
        state_dict = torch.load(args.model, map_location='cpu')
    else:
        state_dict = torch.load(args.model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(new_state_dict)
    else:
        net.cpu()
        net.load_state_dict(new_state_dict)
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = cv2.imread(fn)
        # img = img[:,:, np.newaxis]
        img = img.astype(np.float32)


        mask = predict_img(net=net,
                           full_img=img,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu,
                           channels=out_channels)

        # if args.viz:
        #     print("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask.astype(np.uint8) * 255
            img = img.astype(np.uint8)
            img = cv2.bitwise_and(img, img, mask=result)
            cv2.imwrite(out_files[i], img)

            print("Mask saved to {}".format(out_files[i]))
