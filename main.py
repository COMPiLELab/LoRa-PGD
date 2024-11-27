from robustbench.data import load_cifar10, load_imagenet
from robustbench.utils import load_model, clean_accuracy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import warnings
import math
import utils
from lora_pgd import LoRa_PGD, PGDL2_proj_r
import torchattacks

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn

import sys
sys.path.insert(0, '..')

from utils import imshow

dataset = 'cifar10'
device = 'cuda:0'
data_dir = './data'
model_dir='./models'


if dataset == 'cifar10':
    images, labels = load_cifar10(n_examples=5000, data_dir=data_dir)
    model_list = ['Standard', 'Wang2023Better_WRN-28-10', 'Rebuffi2021Fixing_28_10_cutmix_ddpm', 'Augustin2020Adversarial_34_10_extra', 'Rice2020Overfitting']
    full_rank = 32
    epsilons = [0.5]
elif dataset == 'imagenet':
    images, labels = load_imagenet(n_examples=5000, data_dir=data_dir)
    model_list = ['Standard_R50', 'Wong2020Fast', 'Liu2023Comprehensive_ConvNeXt-L', 'Engstrom2019Robustness','Salman2020Do_R50']
    full_rank = 224
    epsilons = [0.25]
print('[Data loaded]')


steps_i = 10
alphas = 1.8 * np.divide(epsilons, steps_i)
ranks = np.round(np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * full_rank)
ranks = ranks.astype(np.int64)

batch_size = 100
n_batches = math.ceil(images.shape[0] / batch_size)

idx=1
    
results_acc = np.zeros([len(model_list), 5, len(epsilons), len(ranks)])
results_time = np.zeros([len(model_list), 3, len(epsilons), len(ranks)])
figure1 = dict()
figure2 = dict()
fgsm_init = dict()

print_image = True #Change to True to produce images for Figure 3.

for i_m, model_name in enumerate(model_list):
    print('Model: {}'.format(model_name))

    if dataset == 'cifar10':
        model = load_model(model_name, norm='L2', model_dir=model_dir).to(device=device)
    elif dataset == 'imagenet':
        model = load_model(model_name, dataset='imagenet', norm='Linf', model_dir=model_dir).to(device=device)

    print('Clean accuracy')

    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(acc))
    if print_image:
        imshow(images[idx:idx+1], title='Original', title_f='original')

    #if model_name in ['Standard']: #Current mode: Transfer. Add model names to the list to change mode to Warm-up
    atk = torchattacks.PGDL2(model, eps=epsilons[-1], alpha=epsilons[-1], steps=1, random_start=False)
    adv_images, _ = utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device)
    start_init_pgd = (adv_images.to('cpu') - images.to('cpu'))
    u_uap, s_uap, v_uap = torch.linalg.svd(start_init_pgd, full_matrices=False)
    u_uap = u_uap @ torch.diag_embed(torch.sqrt(s_uap))
    v_uap = torch.diag_embed(torch.sqrt(s_uap)) @ v_uap
    start_init_uv = (u_uap, v_uap)
    fgsm_init[model_name] = start_init_uv

    print('Robust accuracies')

    for i_e, eps_i in enumerate(epsilons):

        print('--------------------------------')

        print("PGD L2")
        atk = torchattacks.PGDL2(model, eps=eps_i, alpha=alphas[i_e], steps=steps_i, random_start=False)
        if dataset == 'imagenet':
            atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        adv_images, time_avg= utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device)
        
        pert_res = (adv_images.to('cpu') - images.to('cpu'))
        pert_norms = torch.norm(pert_res, p='nuc', dim=(2, 3))
        pert_norms_l2 = torch.mean(pert_norms)
        figure2[model_name] = pert_norms_l2
        if print_image:
            imshow(adv_images[idx:idx+1], title=r'PGD, $\|\bullet\|_2=$'+str(eps_i)+r', $\|\bullet\|_*=$'+str(np.round(pert_norms_l2.numpy(), decimals=2)), title_f='pgdl2')
        acc = clean_accuracy(model, adv_images.to(device), labels.to(device))

        results_acc[i_m, 0, i_e, 0] = acc
        results_time[i_m, 0, i_e, 0] = time_avg*1./n_batches

        im_0u, im_0s, im_0v = torch.linalg.svd(images,full_matrices=False)
        im_1u, im_1s, im_1v = torch.linalg.svd(adv_images,full_matrices=False)
        s_diff = torch.abs(im_1s.to('cpu') - im_0s.to('cpu'))/im_0s.to('cpu')
        s_avg2 = torch.mean(s_diff, dim=(0, 1))
        figure1[model_name] = s_avg2

        print('- Robust Acc: {} / ({} ms)'.format(acc, time_avg*1./n_batches))
        print('--------------------------------')
        for i_r, rank_i in enumerate(ranks):
            
            print("LoRa_PGD L2", rank_i)
            atk = LoRa_PGD(model, eps=eps_i, rank=rank_i, steps=steps_i, init='lora')
            if dataset == 'imagenet':
                atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_images, time_avg = utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device)

            pert_res = (adv_images.to('cpu') - images.to('cpu'))
            pert_norms = torch.norm(pert_res, p='nuc', dim=(2, 3))
            pert_norms_uv = torch.mean(pert_norms)
            pert_norms_rel = pert_norms_uv/pert_norms_l2
            if print_image:
                imshow(adv_images[idx:idx+1], title=r'LoRa-PGD, $\|\cdot\|_2=$'+str(eps_i)+r', $r=$'+ranks[i_r], title_f='lr_r'+str(rank_i))
            acc = clean_accuracy(model, adv_images, labels)

            results_acc[i_m, 1, i_e, i_r] = acc
            results_time[i_m, 1, i_e, i_r] = time_avg*1./ n_batches
            print('- Robust Acc: {} / ({} ms)'.format(acc, time_avg*1./ n_batches))

            print('--------------------------------')

            print("LoRa_PGD FGSM init (Transfer)", rank_i)
            atk = LoRa_PGD(model, eps=eps_i, rank=rank_i, steps=steps_i, init='fgsm')
            if dataset == 'imagenet':
                atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_images, _= utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device, start_init=fgsm_init['Standard'])
            if print_image:
                imshow(adv_images[idx:idx+1], title=r'LoRa-PGD, $\|\cdot\|_2=$'+str(eps_i)+r', $r=$'+ranks[i_r])
            acc = clean_accuracy(model, adv_images, labels)

            results_acc[i_m, 2, i_e, i_r] = acc
            print('- Robust Acc: {})'.format(acc))

            print('--------------------------------')
            print("LoRa_PGD FGSM init (Warm-up)", rank_i)
            atk = LoRa_PGD(model, eps=eps_i, rank=rank_i, steps=steps_i, init='fgsm')
            if dataset == 'imagenet':
                atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_images, _= utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device, start_init=fgsm_init[model_name])
            if print_image:
                imshow(adv_images[idx:idx+1], title=r'LoRa-PGD, $\|\cdot\|_2=$'+str(eps_i)+r', $r=$'+ranks[i_r], title_f='lr_nuc_r'+str(rank_i))
            acc = clean_accuracy(model, adv_images, labels)

            results_acc[i_m, 3, i_e, i_r] = acc
            print('- Robust Acc: {})'.format(acc))

            print('--------------------------------')

            print("LoRa_PGD Nuclear", rank_i)
            eps_aug = 1./pert_norms_rel
            atk = LoRa_PGD(model,  eps=eps_i*eps_aug, rank=rank_i, steps=steps_i, init='lora')
            if dataset == 'imagenet':
                atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_images, _ = utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device)

            pert_res = (adv_images.to('cpu') - images.to('cpu'))
            pert_norms = torch.norm(pert_res, p='nuc', dim=(2, 3))
            pert_norms_uv_n = torch.mean(pert_norms)
            if print_image:
                imshow(adv_images[idx:idx+1], title=r'LoRa-PGD, $\|\cdot\|_*=$'+str(np.round(pert_norms_uv_n.numpy(), decimals=2)) +', $r=$' + ranks[i_r])
            acc = clean_accuracy(model, adv_images, labels)
            results_acc[i_m, 4, i_e, i_r] = acc
            print('- Robust Acc: {})'.format(acc))
            print('--------------------------------')

            print("Time for PGD projected to r=", rank_i)
            atk = PGDL2_proj_r(model, eps=eps_i, alpha=alphas[i_e], steps=steps_i, random_start=False, rank=rank_i)
            if dataset == 'imagenet':
                atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            adv_images, time_avg = utils.get_adv_images_time(images, labels, atk, batch_size=100, device=device)
        
            results_time[i_m, 2, i_e, i_r] = time_avg*1./ n_batches
            print('- Time ({} ms)'.format(time_avg*1./ n_batches))
            print('--------------------------------')

print('acc', np.array2string(results_acc, separator=', '))
torch.save(results_acc, 'accuracies.pth')
print('time', np.array2string(results_time, separator=', '))
torch.save(results_time, 'time.pth')

if dataset == 'cifar10':
    n_bins = 32
    models_lab = ['WideResNet-28-10', 'Wang23', 'Rebuffi21', 'Augustin20', 'Rice20']
elif dataset == 'imagenet':
    n_bins = 224
    models_lab =['Resnet-50', 'Wong20', 'Liu23', 'Engstrom19','Salman20']
    
fig, axs = plt.subplots(1, 5, sharey=True,  figsize = (15, 5))#tight_layout=True,
fig.subplots_adjust(wspace=0)
for i_m in range(len(model_list)):
    sing_hists = figure1[model_list[i_m]].numpy()
    sing_hists = sing_hists/np.sum(sing_hists)
    axs[i_m].bar(np.arange(n_bins), sing_hists, color = "green", edgecolor="green")#bins=n_bins)
    axs[i_m].tick_params(labelsize=20)
    axs[i_m].set_title(models_lab[i_m], fontsize=20)
axs[2].set_xlabel("#singular value", fontsize=25)
plt.savefig('rel_sing'+dataset+'.png', bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(1, 1, tight_layout=True, figsize = (7, 5))
colors = cm.rainbow(np.linspace(0, 1, 5))
nuc_vals = list(figure2.values())
axs.bar(models_lab, nuc_vals, color=colors)
axs.tick_params(labelsize=17)
axs.set_title(dataset, fontsize=20)
plt.savefig('nuc_bar'+dataset+'.png', bbox_inches='tight')
plt.show()


