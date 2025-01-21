import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import ot
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import average_precision_score
from scipy.stats import wasserstein_distance, ttest_ind

from base.datasets import load_data
from base.networks import MLP


class Trainer:

    def __init__(self, args):
        torch.set_num_threads(8)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """ directories create"""
        os.makedirs(args.model_dir, exist_ok=True)
        self.f_model_dir = args.model_dir + f'{args.source}~{args.target}/model/{args.alg}/{args.seed}/'
        os.makedirs(self.f_model_dir, exist_ok=True)
        
        os.makedirs(args.result_dir, exist_ok=True)
        self.result_dir = args.result_dir + f'{args.source}~{args.target}/'
        os.makedirs(self.result_dir, exist_ok=True)

        self.f_model_path = self.f_model_dir + f'lmda_f-{args.lmda_f}.pt'
        self.is_done = os.path.exists(self.f_model_path)

        if self.is_done:
            print('Already done!')
            return
        pass


    def load_data(self, args):

        """ data load """
        _, source_dloaders, _, target_dloaders, (self.input_dim, self.train_source_size, self.train_target_size) = load_data(seed=args.seed, source=args.source, target=args.target, batch_size=args.batch_size)
        self.trainloader_source, self.trainevalloader_source, self.valloader_source, self.testloader_source = source_dloaders
        self.trainloader_target, self.trainevalloader_target, self.valloader_target, self.testloader_target = target_dloaders


    def _evaluate(self, model, env):

        if env == 'train':
            loader_source = self.trainevalloader_source
            loader_target = self.trainevalloader_target
        elif env == 'val':
            loader_source = self.valloader_source
            loader_target = self.valloader_target
        elif env == 'test':
            loader_source = self.testloader_source
            loader_target = self.testloader_target

        model.eval()
        
        # source
        source_inputs, source_reps = [], []
        source_preds, source_labels, source_sensitives = [], [], []
        with torch.no_grad():
            for inputs, labels, _ in loader_source:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.view(-1, self.input_dim)
                source_inputs.append(inputs)
                source_reps.append(model.extractor(torch.cat([inputs, torch.zeros(inputs.size(0)).reshape(-1, 1).cuda()], dim=1)).detach())
                source_preds.append(model(torch.cat([inputs, torch.zeros(inputs.size(0)).reshape(-1, 1).cuda()], dim=1)).detach())
                source_labels.append(labels)
        source_inputs = torch.cat(source_inputs)
        source_reps = torch.cat(source_reps)
        source_preds, source_labels = torch.cat(source_preds), torch.cat(source_labels)
        source_sensitives = torch.zeros_like(source_labels)
        source_probs = torch.softmax(source_preds, dim=1)[:, 1].flatten()
        source_preds = torch.argmax(source_preds, dim=1).float()

        # target
        target_inputs, target_reps = [], []
        target_preds, target_labels, target_sensitives = [], [], []
        with torch.no_grad():
            for inputs, labels, _ in loader_target:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.view(-1, self.input_dim)
                target_inputs.append(inputs)
                target_reps.append(model.extractor(torch.cat([inputs, torch.ones(inputs.size(0)).reshape(-1, 1).cuda()], dim=1)).detach())
                target_preds.append(model(torch.cat([inputs, torch.ones(inputs.size(0)).reshape(-1, 1).cuda()], dim=1)).detach())
                target_labels.append(labels)
        target_inputs = torch.cat(target_inputs)
        target_reps = torch.cat(target_reps)
        target_preds, target_labels = torch.cat(target_preds), torch.cat(target_labels)
        target_sensitives = torch.ones_like(target_labels)
        target_probs = torch.softmax(target_preds, dim=1)[:, 1].flatten()
        target_preds = torch.argmax(target_preds, dim=1).float()
        
        # all
        all_inputs = torch.cat([source_inputs, target_inputs])
        all_reps = torch.cat([source_reps, target_reps])
        all_preds = torch.cat([source_preds, target_preds])
        all_probs = torch.cat([source_probs, target_probs])
        all_labels = torch.cat([source_labels, target_labels])
        all_sensitives = torch.cat([source_sensitives, target_sensitives])
        
        return all_reps, all_preds, all_probs, all_labels, all_sensitives


    def _get_performances(self, all_reps, all_preds, all_probs, all_labels, all_sensitives):
        assert (all_probs.min() >= 0.0) and (all_probs.max() <= 1.0)
        
        """ task performances """
        # acc
        acc = (all_preds == all_labels).float().mean()
        # bacc
        bacc = (all_preds[all_labels == 0] == all_labels[all_labels == 0]).float().mean()
        bacc += (all_preds[all_labels == 1] == all_labels[all_labels == 1]).float().mean()
        bacc /= 2.0
        # ap
        ap = average_precision_score(all_labels.detach().cpu().numpy(), all_probs.detach().cpu().numpy())

        preds0, preds1 = all_preds[all_sensitives == 0], all_preds[all_sensitives == 1]
        probs0, probs1 = all_probs[all_sensitives == 0], all_probs[all_sensitives == 1]
        
        """ fairness performances """
        # dp
        dp = (preds0.mean() - preds1.mean()).abs()
        # meandp
        meandp = (probs0.mean() - probs1.mean()).abs()
        # wasserstein dp (assumed as Gaussian)
        wdp = wasserstein_distance(probs0.detach().cpu().numpy(), probs1.detach().cpu().numpy())
        # sdp & ksdp
        sdps = []
        for tau in np.linspace(0.1, 1.0, 10):
            sdps.append(
                ( (probs0 > tau).float().mean() - (probs1 > tau).float().mean() ).abs().item()
                )
        sdp = torch.tensor([float(np.mean(sdps))])
        ksdp = torch.tensor([float(np.max(sdps))])

        """ all performances"""
        task_ = (round(acc.item(), 4), round(bacc.item(), 4), round(ap.item(), 4))
        fair_ = (round(dp.item(), 4), round(meandp.item(), 4), round(wdp.item(), 4), round(sdp.item(), 4), round(ksdp.item(), 4))

        return task_, fair_


    """ main functions """


    def _train_single_epoch_f(self, single_epoch_done, model, optimizer, scheduler, criterion, lmda_f, epoch_losses):
        source_iter, target_iter = iter(self.trainloader_source), iter(self.trainloader_target)
        losses = deepcopy(epoch_losses)
        cnt = 0
        while not single_epoch_done:
            try:
                source_input, source_label, _ = next(source_iter)
                single_epoch_done = False
            except:
                source_iter = iter(self.trainloader_source)
                source_input, source_label, _ = next(source_iter)
                single_epoch_done = True
            
            try:
                target_input, target_label, _ = next(target_iter)
                single_epoch_done = False
            except:
                target_iter = iter(self.trainloader_target)
                target_input, target_label, _ = next(target_iter)
                single_epoch_done = True

            source_input, source_label = source_input.cuda(), source_label.cuda()
            target_input, target_label = target_input.cuda(), target_label.cuda()
            # transporting
            source_weight, target_weight = torch.ones(size=(source_input.shape[0], )) / source_input.shape[0], torch.ones(size=(target_input.shape[0], )) / target_input.shape[0]
            M_source_target = ot.dist(source_input, target_input)
            G_source_target = ot.emd(source_weight, target_weight, M_source_target)
            mappedsource_input = target_input[torch.argmax(G_source_target, dim=1)]
            del M_source_target; del G_source_target

            source_logits = model(torch.cat([source_input, torch.zeros(source_input.size(0)).reshape(-1, 1).cuda()], dim=1))
            target_logits = model(torch.cat([target_input, torch.ones(target_input.size(0)).reshape(-1, 1).cuda()], dim=1))
            source_probs = torch.softmax(source_logits, dim=1)[:, 1].flatten()
            target_probs = torch.softmax(target_logits, dim=1)[:, 1].flatten()
            source_label = source_label.flatten().to(torch.long)
            target_label = target_label.flatten().to(torch.long)

            TASK_loss_source = criterion(source_logits, source_label)
            TASK_loss_target = criterion(target_logits, target_label)
            source_weight = self.train_source_size / (self.train_source_size + self.train_target_size)
            target_weight = 1.0 - source_weight
            TASK_loss = source_weight * TASK_loss_source + target_weight * TASK_loss_target

            # fair loss
            mappedsource_logits = model(torch.cat([mappedsource_input, torch.ones(mappedsource_input.size(0)).reshape(-1, 1).cuda()], dim=1))
            mappedsource_probs = torch.softmax(mappedsource_logits, dim=1)[:, 1].flatten()
            FAIR_loss = (source_probs - mappedsource_probs).abs().mean()

            loss = TASK_loss + lmda_f * FAIR_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (scheduler != None) and single_epoch_done:
                scheduler.step()
            
            cnt += len(source_input)
            losses['TASK'] += TASK_loss_source.item() * len(source_input) + TASK_loss_target.item() * len(target_input)
            losses['FAIR'] += FAIR_loss.item() * len(source_input)

        epoch_losses['TASK'] += losses['TASK'] / (2 * cnt)
        epoch_losses['FAIR'] += losses['FAIR'] / cnt

        return single_epoch_done, epoch_losses


    def train_f(self, args):

        self.f_model_path = self.f_model_dir + f'lmda_f-{args.lmda_f}.pt'

        criterion = F.cross_entropy

        model = MLP(input_dim=self.input_dim+1, hidden_dims=[self.input_dim, self.input_dim], output_dim=2, act='ReLU').cuda()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
        model_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=model_optimizer,
                                                            lr_lambda=lambda epoch: 0.95**epoch)

        for epoch in range(args.epochs):
            epoch += 1
            model.train()
            epoch_losses = {'TASK': 0.0, 'FAIR': 0.0}
            single_epoch_done = False
            single_epoch_done, epoch_losses = self._train_single_epoch_f(single_epoch_done=single_epoch_done, model=model,
                                                                         optimizer=model_optimizer, scheduler=model_scheduler,
                                                                         criterion=criterion, 
                                                                         lmda_f=args.lmda_f, epoch_losses=epoch_losses)
            print(f'[{epoch}/{args.epochs}] TASK loss: {round(epoch_losses["TASK"], 5)}, FAIR loss: {round(epoch_losses["FAIR"], 5)}')

            # evaluation
            all_reps, all_preds, all_probs, all_labels, all_sensitives = self._evaluate(model=model, env=args.eval_env)
            task_, fair_ = self._get_performances(all_reps, all_preds, all_probs, all_labels, all_sensitives)
            acc, bacc, ap = task_
            dp, meandp, wdp, sdp, ksdp = fair_
            print(f'\t acc = {acc}, dp = {dp}, meandp = {meandp}, wdp = {wdp}')

            results = {f'{args.eval_env} acc': acc,
                       f'{args.eval_env} bacc': bacc,
                       f'{args.eval_env} ap': ap,
                       f'{args.eval_env} dp': dp,
                       f'{args.eval_env} meandp': meandp,
                       f'{args.eval_env} wdp': wdp,
                       f'{args.eval_env} sdp': sdp,
                       f'{args.eval_env} ksdp': ksdp
                       }

        # result save
        result_name = self.result_dir + 'result.csv'
        if os.path.exists(result_name):
            file = open(result_name, 'a', newline='')
            writer = csv.writer(file)
        else:
            file = open(result_name, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(['alg', 'lmda_f', 'seed'] + list(results.keys()))
        writer.writerow([f'{args.alg}', f'{args.lmda_f}', f'{args.seed}'] + list(results.values()))
        file.close()

        torch.save(model.state_dict(), self.f_model_path)

