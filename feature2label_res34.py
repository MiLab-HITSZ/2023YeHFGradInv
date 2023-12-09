# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/12 2:33 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : feature_analyze.py

import torch
from custom_dataset import CustomData
import breaching

def get_shared():
    cfg = breaching.get_config(overrides=["case=12_custom_imagenet", "attack=our_res34_pretrained"])

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    # print(setup)

    cfg.case.data.partition = 'balanced'  # 'balanced'
    cfg.case.user.user_idx  = 0
    cfg.case.model = "resnet34"  # "resnet34" #

    cfg.case.user.provide_labels=True # This attack can reconstruct label information via optimization.
    cfg.case.user.provide_buffers=False
    cfg.case.server.provide_public_buffers=True
    cfg.case.server.pretrained=True

    cfg.case.data.mix = None
    only_mix = True
    using_mix_defense = False
    if using_mix_defense:
        cfg.case.data.mix = 'mixup'
        only_mix = True

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker_loss = torch.nn.CrossEntropyLoss()
    attacker = breaching.attacks.prepare_attack(server.model, attacker_loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    server_payload = server.distribute_payload()

    cus_data = CustomData(data_dir='custom_data/aweb_img/', dataset_name='ImageNet', case='12_custom_imagenet', mix=cfg.case.data.mix, only_mix=only_mix)
    shared_data, true_user_data = user.compute_local_updates(server_payload, custom_data=cus_data.process_data())
    return shared_data, true_user_data

def get_initpop(label_list, n_data, n_pop):
    nc = len(label_list)
    pop = torch.randint(1,n_data-nc+2, (n_pop,nc))
    for i in range(n_pop):
        tmp = pop[i].sum()/n_data
        pop[i] = (pop[i]/tmp).floor()
        pop[i] = torch.max(torch.ones_like(pop[i]), pop[i])
        pop[i,-1] = n_data-pop[i,:-1].sum()
    return pop

def rescale(tensor, n_data):
    tmp = tensor.sum()/n_data
    tensor = (tensor/tmp).floor()
    tensor = torch.max(torch.ones_like(tensor), tensor)
    tensor[-1] = n_data-tensor[:-1].sum()
    return tensor

def crossover(par1, par2, n_data):
    p = 0.5
    mask = torch.rand(par1.shape) < p
    child = mask * par1 + (~mask) * par2
    child = rescale(child, n_data)
    return child

def fit_ness(tensor, avg, gc_sort, n_data):
    vec = [i*avg/n_data for i in tensor]
    v = torch.cat(vec, dim=1)
    scalar_product = (v * gc_sort).sum()
    v_norm = v.norm(2)
    g_norm = gc_sort.norm(2)
    fit = scalar_product / v_norm / g_norm
    return fit

def mutation(tensor):
    idx = torch.randperm(len(tensor))
    for i in range(len(tensor)-1):
        if tensor[idx[i]] > 1:
            tensor[idx[i]] -= 1
            break
    try:
        tensor[idx[i+1]] += 1
    except:
        tensor[idx[i-1]] += 1
    return tensor

def get_child_pop(par_pop, n_data):
    n_pop = par_pop.shape[0]
    kk = torch.randperm(n_pop)
    child_pop = par_pop[kk]
    for i in range(n_pop):
        child_pop[i] = crossover(par_pop[i], child_pop[i], n_data)
        child_pop[i] = mutation(child_pop[i])
    return child_pop

def selection(par_pop, child_pop, avg, gc_sort, n_data):
    n_pop = par_pop.shape[0]
    fit_all = []
    for i in range(n_pop):
        fit_par = fit_ness(par_pop[i], avg, gc_sort, n_data)
        fit_chi = fit_ness(child_pop[i], avg, gc_sort, n_data)
        par_pop[i] = par_pop[i] if fit_par >= fit_chi else child_pop[i]
        fit_all.append(torch.max(fit_par, fit_chi).item())
    return par_pop, fit_all

def run(init_pop, iters, avg, gc_sort, n_data):
    par_pop = init_pop
    for i in range(iters):
        child_pop = get_child_pop(par_pop, n_data)
        par_pop, fit = selection(par_pop, child_pop, avg, gc_sort, n_data)
        m_index = fit.index(max(fit))
        # print(f'best individual in iters-{i}: {par_pop[m_index]}, fitness: {max(fit)}')
    nums = par_pop[m_index]
    return nums

def get_labels(shared_data, num_data_points, n_pop, iters):
    ref = torch.load('ref_res34.pt')  #ref_res34.pt
    idx = ref['sort_var'][0:20]
    avg = ref['avg'][:,idx]
    g_fc = shared_data["gradients"][-2].cpu()
    g_mean = g_fc.mean(dim=1)
    label_list = []
    for i in range(1000):
        if g_mean[i] < 0:
            label_list.append(i)
    g_cut = g_fc[label_list]
    gc_sort = torch.sort(torch.abs(g_cut), dim=1)[0][:,idx].reshape(1,-1)
    n_data = num_data_points
    pop = get_initpop(label_list, n_data, n_pop)
    nums = run(pop, iters, avg, gc_sort, n_data)
    labels = []
    for i,j in zip(label_list, nums):
        labels += [i]*j.item()
    labels = torch.tensor(labels).reshape(-1).cuda()
    return labels



