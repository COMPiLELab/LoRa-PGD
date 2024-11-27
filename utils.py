import torch
import math
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def get_adv_images_time(images, labels, attack, start_init=None, batch_size=100, device='cpu'):
    n_batches = math.ceil(images.shape[0] / batch_size)
    adv_images = torch.zeros_like(images).to(device)
    time_avg = 0.
    for counter in range(n_batches):
        x_curr = images[counter * batch_size:(counter + 1) * batch_size].to(device)
        y_curr = labels[counter * batch_size:(counter + 1) * batch_size].to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if start_init != None:
            start_u,start_v = start_init
            start_init_uv=(start_u[counter * batch_size:(counter + 1) * batch_size], start_v[counter * batch_size:(counter + 1) * batch_size])
            start.record()
            adv_images[counter * batch_size:(counter + 1) * batch_size] = attack(x_curr, y_curr, start_init=start_init_uv)
            end.record()
        else:
            start.record()
            adv_images[counter * batch_size:(counter + 1) * batch_size] = attack(x_curr, y_curr)
            end.record()
        torch.cuda.synchronize()
        time_avg += start.elapsed_time(end)
    return adv_images, time_avg

def imshow(img, title, title_f=None):
    #from torchattacks modified to save file
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.savefig(title_f + '.png', bbox_inches='tight')
    plt.show()

