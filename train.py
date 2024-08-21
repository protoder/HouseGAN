import argparse
import os
import numpy as np
import math
import random

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn, train_collate_fn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
from misc.utils import combine_images, _init_input, _init_input_predict, selectRandomNodes, selectNodesTypes
from models.models import Discriminator, Generator, compute_gradient_penalty
import multiprocessing
from tqdm import tqdm
import warnings
# Устанавливаем уровень фильтрации предупреждений на "ignore"
warnings.filterwarnings("ignore")




'''parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10 help="size of the batches")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between image sampling")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--target_set", type=int, default=8, choices=[5, 6, 7, 8], help="which split to remove")
parser.add_argument("--data_path", type=str, default='', help="path to the dataset")
parser.add_argument("--test_path", type=str, default='rplan_test/', help="path to the dataset")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda for gradient penalty")
opt = parser.parse_args()
'''


class parser:
    def __init__(self):
        self.n_epochs=100000
        self.batch_size=256
        self.g_lr=0.0001
        self.d_lr=0.0001
        self.b1=0.5
        self.b2=0.999
        self.n_cpu=2
        self.sample_interval=100
        self.exp_folder = 'exp'
        self.n_critic=2
        self.target_set=8
        self.data_path='data/' #'rplan/'
        self.lambda_gp=10
        self.test_path='rplan_test/'
        self.checkpoint ='checkpoints/pretrained.pth'
opt = parser()
TestBatches = 30
torch.manual_seed(20)
random.seed(20)

exp_folder = "{}_{}".format(opt.exp_folder, opt.target_set)
os.makedirs("./exps/"+exp_folder, exist_ok=True)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
distance_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

#model = Generator()
#generator.load_state_dict(torch.load(opt.checkpoint, map_location=torch.device('cpu')), strict=True)
#generator = generator.eval()


# Visualize a single batch
def visualizeSingleBatch(fp_loader_test, opt, exp_folder, batches_done, batch_size=8, draw=False):
    #print('Loading saved model ... \n{}'.format('{}checkpoints/{}_{}.pth'.format(CrPath, exp_folder, batches_done)))
    generatorTest = Generator()
    generatorTest.load_state_dict(torch.load('{}checkpoints/{}_{}.pth'.format(CrPath, exp_folder, batches_done)))
    generatorTest = generatorTest.eval()

    if torch.cuda.is_available():
        generatorTest.cuda()
    else:
        generatorTest.cpu()

    with torch.no_grad():
        # Unpack batch
        mks, nds, eds, nd_to_sample, ed_to_sample = next(iter(fp_loader_test))
        real_mks = Variable(mks.type(Tensor))
        given_nds = Variable(nds.type(Tensor))
        given_eds = eds
        # Select random nodes

        ind = mks == 1
        gen_mks = mks.clone().detach()
        gen_mks[ind] = -1
        gen_mks = 2 * (gen_mks + 0.5)

        given_masks = gen_mks.unsqueeze(1)
        inds_masks = torch.ones_like(given_masks)

        shifted_array = np.roll(nd_to_sample, -1)

        # Получаем индексы первого вхождения каждой цифры
        first_occurrences = np.array(nd_to_sample) != shifted_array
        batchStartes = np.arange(len(nd_to_sample))[first_occurrences] + 1
        batchStartes = np.insert(batchStartes, 0, 0)
        batchSizes = np.diff(batchStartes)

        batchStartes = torch.tensor(batchStartes).to(device)
        batchSizes = torch.tensor(batchSizes).to(device)

        if torch.cuda.is_available():
            given_masks = torch.cat([given_masks, inds_masks], 1).cuda()
            given_nds = given_nds.float().cuda()
            given_eds = torch.tensor(given_eds).long().cuda()
            z = torch.randn(len(given_nds), 128).float().cuda()
        else:
            given_masks = torch.cat([given_masks, inds_masks], 1).cpu()
            given_nds = given_nds.float().cpu()
            given_eds = torch.tensor(given_eds).long().cpu()
            z = torch.randn(len(given_nds), 128).float().cpu()

        gen_mks = generatorTest(z, given_masks, given_nds, given_eds, batchStartes=batchStartes[:-1],
                                batchSizes=batchSizes)

        if draw:
            # Generate image tensors
            real_imgs_tensor = combine_images(real_mks, given_nds, given_eds, \
                                              nd_to_sample, ed_to_sample)
            fake_imgs_tensor = combine_images(gen_mks, given_nds, given_eds, \
                                              nd_to_sample, ed_to_sample)
            # Save images
            save_image(real_imgs_tensor, "{}exps/{}/{}_real.png".format(CrPath, exp_folder, batches_done), \
                       nrow=12, normalize=False)
            save_image(fake_imgs_tensor, "{}exps/{}/{}_fake.png".format(CrPath, exp_folder, batches_done), \
                       nrow=12, normalize=False)

        err = distance_loss(gen_mks, given_masks[:, 0, :, :]) * 1000 \
            if len(ind_fixed_nodes) > 0 else torch.tensor(0.0)

    return err
import matplotlib.pyplot as plt
def plot_training_results(results, save_path):
    if len(results) < 5:
        return

    results = np.log10(results)
    epochs = range(0, len(results))

    train_errors = results[:, 0]
    test_errors = results[:, 1]

    plt.plot(epochs, train_errors, 'b', label='Training error')
    plt.plot(epochs, test_errors, 'r', label='Test error')
    plt.title('Training and Test Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def Start(CrPath):
    global generator
    global discriminator
    opt.data_path = CrPath + opt.data_path
    opt.test_path =  CrPath + opt.test_path
    opt.checkpoint =  CrPath + opt.checkpoint
    # Configure data loader
    fp_dataset_train = FloorplanGraphDataset(opt.data_path, transforms.Normalize(mean=[0.5], std=[0.5]),
                                             target_set=opt.target_set)
    fp_loader = torch.utils.data.DataLoader(fp_dataset_train,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.n_cpu,
                                            # collate_fn=train_collate_fn,
                                            pin_memory=False)

    '''fp_dataset_test = FloorplanGraphDataset(opt.test_path, transforms.Normalize(mean=[0.5], std=[0.5]),
                                            target_set=opt.target_set, split='eval')
    fp_loader_test = torch.utils.data.DataLoader(fp_dataset_test,
                                                 batch_size=8,
                                                 shuffle=True,
                                                 num_workers=opt.n_cpu,
                                                 collate_fn=floorplan_collate_fn,
                                                 pin_memory=False)'''
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    batches_done = 0

    import os.path

    current_index = '_j'#1351

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if os.path.exists(f"{CrPath}checkpoints/current{current_index}.pth"):

        try:
            dict = torch.load(f"{CrPath}checkpoints/current{current_index}.pth", map_location=torch.device(device))
            discriminator.load_state_dict(dict['d'])
            discriminator.to(device)
            optimizer_D.load_state_dict(dict['od'])
        except Exception as e:
            print(f"Не удалось прочитать дискриминатор: {e}")

        try:
            generator.load_state_dict(dict['g'])
            generator.to(device)
            optimizer_G.load_state_dict((dict['og']))
        except Exception as e:
            print(f"Не удалось прочитать генератор: {e}")

        try:
            results = dict['stat'][:current_index+1]
        except:
            results = dict['stat']

        # generator = generator.eval()
        discriminator = discriminator.train()

        crEpoch = len(results)

        print('Сеанс восстановлен')

    else:
        crEpoch = 0
        results = []

    generator = generator.train()



    generator.to(device)
    discriminator.to(device)
    # adversarial_loss.to(device)


    optimizer_D.zero_grad()

    criterion = nn.BCEWithLogitsLoss()

    size = len(fp_loader)
    testStart = 0  # size * 99 // 100

    for p in discriminator.parameters():
        p.requires_grad = True

    # shufle_size = (testStart + 1) * opt.batch_size
    # shufle_indexes = np.arange(shufle_size)

    for p in generator.parameters():
        p.requires_grad = True

    for epoch in range(crEpoch, opt.n_epochs):
        # w = torch.load('checkpoints/current_g235.pth', map_location=torch.device('cpu'))
        # generator.load_state_dict(w)
        # generator.eval()
        res = np.zeros(5)
        sumErr = testErr = d_loss_sum = gradient_penalty_sum = sum_among = 0
        counts = 1

        for i, (mks, nds) in enumerate(t := tqdm(fp_loader)):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_sz = len(mks)
            optimizer_D.zero_grad()
            # Select random nodes
            z = torch.randn(batch_sz, 128).float().to(device)
            given_nds = nds.to(device)
            train_mks = mks[:, 0:1].to(device)
            '''
                Шум
                картинка (-1 - пустое пространство, 0 - наружа (только в моей реализации), 1 - объект )
                OHE тип комнаты или двери
                соседство комнат ( [Номер комнаты1, флаг, Номер комнаты 2]. -1 если не соседствуют, 1 - если да)

            '''

            # Генератор получает на вход картинки с учетом информации о блокировке, в формате [N, 2, :, :]
            gen_mks = generator(z, train_mks, given_nds)

            # дискриминатор информации о блокировках не получает
            # Real images
            real_mks = mks[:, 1:].to(device)

            real_validity = discriminator(real_mks, given_nds)
            # Fake images
            fake_validity = discriminator(gen_mks.detach(), given_nds)

            # Measure discriminator's ability to classify real from generated samples
            gradient_penalty = compute_gradient_penalty(discriminator, real_mks, \
                                                        gen_mks.detach(), given_nds, \
                                                        batch_sz)

            # ошибка дискриминатора
            d_loss = -torch.mean(real_validity)+ torch.mean(fake_validity) \
                     + opt.lambda_gp * gradient_penalty
            # Update discriminator
            d_loss.backward()
            optimizer_D.step()

            # расчет метрики ошибки определения фэйка (идея в том, на сколько сильно пространство оценки фэйков пересекается с оценкой реальных)
            fake = fake_validity.detach()
            real = real_validity.detach()
            among = float((fake >= real.min()).sum() / len(fake))
            sum_among += among

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            '''del d_loss
            del real_validity
            del fake_validity
            del gradient_penalty'''

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:


                d_loss_sum += float(d_loss.detach())
                gradient_penalty_sum += float(gradient_penalty.detach())

                #torch.cuda.empty_cache()

                for p in discriminator.parameters():
                    p.requires_grad = False

                z = Variable(Tensor(np.random.normal(0, 1, tuple((real_mks.shape[0], 128)))))
                gen_mks = generator(z, train_mks, given_nds)
                # Score fake images
                fake_validity = discriminator(gen_mks, given_nds)

                # Compute L1 loss
                # Эта ошибка нужна для фиксации фиксированных комнат
                mask = ((train_mks + 1) / 2)
                gen_mks_boards = gen_mks.mean(1).view(-1, 1, 64, 64)
                err = distance_loss(gen_mks_boards * mask, -mask) * 1000

                # Update generator
                g_loss = -torch.mean(fake_validity) + err

                g_loss.backward()
                # Update optimizer
                optimizer_G.step()

                sumErr += float(g_loss.detach())
                t.set_postfix({'err': sumErr / counts})
                counts += 1

                for p in discriminator.parameters():
                    p.requires_grad = True


                #torch.cuda.empty_cache()

        results.append((sumErr / counts, d_loss_sum / counts, gradient_penalty_sum / counts, sum_among/counts))
        np.save(CrPath + 'res.npy', np.array(results))

        cnt = (i - TestBatches)
        print('\nЭпоха', epoch, sumErr / counts, testErr / (i - testStart), sum_among/counts)

        dict = {'d': discriminator.state_dict(),
                'g': generator.state_dict(),
                'od': optimizer_D.state_dict(),
                'og': optimizer_G.state_dict(),
                'stat': results}
        torch.save(dict, CrPath + 'checkpoints/current' + str(epoch) + '.pth')

        save_path = f"{CrPath}checkpoints/plot{epoch}.jpg"
        plot_training_results(results, save_path)

        '''random.shuffle(shufle_indexes)
        fp_dataset_train.Train[:shufle_size] = fp_dataset_train.Train[shufle_indexes]
        fp_dataset_train.OHE[:shufle_size] = fp_dataset_train.OHE[shufle_indexes]'''


import pickle
# ----------
#  Training
# ----------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    Start('./')



