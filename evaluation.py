import torch
import numpy as np
import os, warnings, logging, glob
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
from ema_pytorch import EMA

from utils import writelog, unnormalize_to_zero_to_one
from Models.diffusion import Diffusion

from metrics.context_fid import Context_FID
from metrics.correlational_score import CrossCorrelLoss
from metrics.discriminative_metric import discriminative_score_metrics
from metrics.predictive_metric import predictive_score_metrics

from metrics.utils import visualization

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-path', '--path', type=str, default='./experiments/')
    parser.add_argument('-date', '--date', type=str, default='0904')
    parser.add_argument('-datasets', '--datasets', type=str, default='sine') # sine, energy, etth, stock
    parser.add_argument('-iter', '--iter', type=int, default=5)
    args = parser.parse_args()
    return args

def random_choice(size, num_select=100):
    select_idx = np.random.randint(low=0, high=size, size=(num_select,))
    return select_idx

def main():
    # parser
    args = parse_args()
    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    f = open(args.path + "{}_log.txt".format(args.datasets), 'a')
    writelog(f, '-'*20)
    writelog(f, 'Evaluation')
    writelog(f, 'Datasets: {}'.format(args.datasets))
    writelog(f, 'Date: {}'.format(args.date))
    if args.datasets == 'sine':
        ori = np.load(args.path + 'Output/samples/sine_ground_truth_24_train.npy')
    else:
        ori = np.load(args.path + 'Output/samples/{}_norm_truth_24_train.npy'.format(args.datasets))

    sampling_dir = args.path + 'Output/{}/{}/'.format(args.datasets, args.date)
    time = sorted(os.listdir(sampling_dir))[-1]
    writelog(f, "Time: {}".format(time))
    writelog(f, '-' * 20)
    for samples in glob.glob(sampling_dir + time + '/*.npy'):
        iter = samples.split('/')[-1].split('_')[1]
        fake = unnormalize_to_zero_to_one(np.load(samples))
        len = min(ori.shape[0], fake.shape[0])
        dis_scores = []
        pre_scores = []
        context_fid = []

        corr_scores = []
        x_real = torch.from_numpy(ori)
        x_fake = torch.from_numpy(fake)
        size = int(x_real.shape[0] / args.iter)
        writelog(f, 'Checkpoint: {}'.format(iter))
        for i in range(args.iter):
            dis_score = discriminative_score_metrics(ori[:len], fake[:len])
            pre_score = predictive_score_metrics(ori[:len], fake[:len])
            fid = Context_FID(ori[:len], fake[:len])

            dis_scores.append(dis_score[0])
            pre_scores.append(pre_score)
            context_fid.append(fid)

            # correlational score
            real_idx = random_choice(x_real.shape[0], size)
            fake_idx = random_choice(x_fake.shape[0], size)
            corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
            loss = corr.compute(x_fake[fake_idx, :, :])
            corr_scores.append(loss.item())

            writelog(f, 'Iteration-{}'.format(i))
            writelog(f, 'Discriminative score: {}, Predictive score: {}'.format(str(dis_score[0]), str(pre_score)))
            writelog(f, 'Context-FID: {}, Correlational score: {}'.format(str(fid), str(loss.item())))
        np.savez('')
        writelog(f, 'Discriminative score (mean/std): {}/{}'.format(str(np.mean(dis_scores)), str(np.std(dis_scores))))
        writelog(f, 'Predictive score (mean/std): {}/{}'.format(str(np.mean(pre_scores)), str(np.std(pre_scores))))
        writelog(f, 'Context-FID score (mean/std): {}/{}'.format(str(np.mean(context_fid)), str(np.std(context_fid))))
        writelog(f, 'Correlational score (mean/std): {}/{}'.format(str(np.mean(corr_scores)), str(np.std(corr_scores))))
        writelog(f, '-' * 20)
    f.close()

if __name__ == "__main__":
    main()

