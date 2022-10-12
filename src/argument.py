import argparse


def parse():
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='The faster proximal algorithm, the better unfolded deep learning architecture ? The study case of image denoising.')

    parser.add_argument('--root_dir'       , type=str, default='../dataset/BSDS500/data/images', help='root of dataset')
    parser.add_argument('--output_dir'     , type=str, default='../checkpoints/', help='directory of saved checkpoints')
    parser.add_argument('--num_epochs'     , type=int, default=2, help='number of epochs')
    parser.add_argument('--K'              , type=int, default=10, help='number of unfolded layer')
    parser.add_argument('--F'              , type=int, default=20, help='number of features')
    parser.add_argument('--model'          , type=str, default='DnCNN',help='DnCNN,unfolded_ISTA,unfolded_FISTA,compare_diff_K,...')
    parser.add_argument('--lr'             , type=float, default=1e-4,help='learning rate')
    parser.add_argument('--image_size'     , type=tuple, default=(180, 180))
    parser.add_argument('--test_image_size', type=tuple, default=(320, 320))
    parser.add_argument('--batch_size'     , type=int, default=6)
    parser.add_argument('--sigma'           , type=int)
    return parser.parse_args()


class Args():
    '''
    For jupyter notebook
    '''

    def __init__(self):
        self.output_dir = '../checkpoints/'
        self.num_epochs = 200
        self.D = 6
        self.C = 64
        self.plot = False
        self.model = 'DnCNN'
        self.lr = 1e-4
        self.batch_size = 5
