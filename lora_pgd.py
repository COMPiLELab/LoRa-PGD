import torch
import torch.nn as nn
import datetime

from torchattacks.attack import Attack

class LoRa_PGD(Attack):
    #written in a way to be torchattacks compatible
    r"""
    Low-rank PGD style attack
    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        rank (int): chosen rank.
        eps (float): maximum perturbation. (Default: 1.0)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)
        init (str): Initialization choice (Default: 'lora')

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(
        self,
        model,
        rank,
        eps=1.0,
        steps=10,
        eps_for_division=1e-10,
        init='lora'
    ):
        super().__init__("lruv", model)
        self.eps = eps
        self.rank = int(rank)
        self.steps = steps
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default"]
        self.init = init

    def forward(self, images, labels, start_init=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss_f = nn.CrossEntropyLoss()

        images.requires_grad = True
        bi_shape = images.shape #[batch, 3, n, n]
        if self.init == 'lora':
            u_im = torch.randn([bi_shape[0], bi_shape[1], bi_shape[2], self.rank], device=self.device)
            norm_u = torch.norm(u_im.view(bi_shape[0], -1), p=2, dim=1)
            u_im = (u_im/(norm_u.view(bi_shape[0], 1, 1, 1))).detach()
            v_im = torch.zeros([bi_shape[0], bi_shape[1], self.rank, bi_shape[3]], device=self.device).detach()
        elif self.init == 'fgsm':
            u_im, v_im = start_init
            u_im = u_im[:, :, :, :self.rank].detach().to(self.device)
            v_im = v_im[:, :, :self.rank, :].detach().to(self.device)

        for _ in range(self.steps):

            u_im.requires_grad = True
            v_im.requires_grad = True

            if self.eps == 0.:
                break
            delta = torch.einsum('bcik,bckj->bcij', u_im, v_im)
            delta_norm =  torch.linalg.vector_norm(delta.reshape(bi_shape[0], -1), ord=2, dim=1)+self.eps_for_division
            im_per = torch.clamp(images + self.eps*delta/delta_norm.view(bi_shape[0], 1, 1, 1), 0, 1)

            output = self.get_logits(im_per)
            loss = loss_f(output, labels)
            data_grad = torch.autograd.grad(loss, inputs=[u_im, v_im], retain_graph=False, create_graph=False)

            data_grad_u = data_grad[0].detach()
            data_grad_v = data_grad[1].detach()
            norm_grad_u =  torch.linalg.vector_norm(data_grad_u.reshape(bi_shape[0], -1), ord=2, dim=1)+self.eps_for_division
            norm_grad_v =  torch.linalg.vector_norm(data_grad_v.reshape(bi_shape[0], -1), ord=2, dim=1)+self.eps_for_division
            u_im = u_im + (data_grad_u/(norm_grad_u.view(bi_shape[0], 1, 1, 1)))
            v_im = v_im + (data_grad_v/(norm_grad_v.view(bi_shape[0], 1, 1, 1)))
            u_im = u_im.detach()
            v_im = v_im.detach()

        delta = torch.einsum('bcik,bckj->bcij', u_im, v_im)
        delta_norm =  torch.linalg.vector_norm(delta.detach().reshape(bi_shape[0], -1), ord=2, dim=1)
        delta = self.eps*delta/(delta_norm.view(bi_shape[0], 1, 1, 1)+self.eps_for_division)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    

class PGDL2_proj_r(Attack):
    #from torchattacks modified to projected on low-rank
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=1.0,
        alpha=0.2,
        steps=10,
        random_start=True,
        eps_for_division=1e-10,
        rank=5,
        p_norm=2
    ):
        super().__init__("PGDL2_cut", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.rank = int(rank)
        self.p_norm = p_norm
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for s_i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad_norms = (
                torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                + self.eps_for_division
            )  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            if s_i == self.steps - 1:
                u_im, s_im, v_im = torch.linalg.svd(delta, full_matrices=False)
                u_cut = u_im[:, :, :, :self.rank]
                s_cut = s_im[:, :, :self.rank]
                v_cut = v_im[:, :, :self.rank, :]
                delta = u_cut@ torch.diag_embed(s_cut)@ v_cut

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images