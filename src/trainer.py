from src.checkpoint import Checkpoint
from src.loss import ELBO, custom_ELBO, flat_ELBO, only_r_loss
from src.plot import plot_pred_and_target

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.autograd import Variable

from math import exp
import numpy as np

from tqdm import tqdm
import wandb
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def decay_old(x):
    return 0.001 + (0.999)*(0.9999)**x

def lr_decay(global_step,
    init_learning_rate = 1e-3,
    min_learning_rate = 1e-5,
    decay_rate = 0.9999):
    # lr = ((init_learning_rate - min_learning_rate) *
    #      pow(decay_rate, global_step) +
    #      min_learning_rate)
    lr = init_learning_rate * pow(-0.9999, global_step)
    return lr

class Trainer:
    def __init__(self, 
                 learning_rate=1e-3,
                 KL_rate=0.9999,
                 free_bits=256,
                 sampling_rate=2000,
                 batch_size=512, 
                 print_every=1000, 
                 checkpoint_every=10000,
                 plot_every=1000,
                 checkpoint_dir='checkpoint',
                 output_dir='outputs',
                 use_danceability=True,
                 use_fake_data=False):
        self.learning_rate = learning_rate
        self.KL_rate = KL_rate
        self.free_bits = free_bits
        self.optimizer=None
        self.scheduler=None
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.print_every = print_every
        self.checkpoint_every = checkpoint_every
        self.plot_every = plot_every
        self.output_dir = output_dir
        self.use_danceability = use_danceability
        self.use_fake_data = use_fake_data

        
    def inverse_sigmoid(self,step):
        """
        Compute teacher forcing probability with inverse sigmoid
        """
        k = self.sampling_rate
        if k == None:
            return 0
        if k == 1.0:
            return 1
        return k/(k + exp(step/k))
        
    def KL_annealing(self, step, start, end):
        return end + (start - end)*(self.KL_rate)**step


    def plot_last_batch(self, model, batch, use_teacher_forcing=True, da=None, num_plots=1, is_eval=True):
        model.eval()
        #pred, mu, sigma, z = model(batch, use_teacher_forcing, da)

        #pred = model.reconstruct(batch, 1)
        #model.train()
        with torch.no_grad():
            #pred = model.reconstruct(batch, 1)
            pred, mu, sigma, z = model(batch, use_teacher_forcing, da)
            batch = batch.detach().cpu()
            print("PLOTTING ------")
            # print(f"pred: {pred.size()}")
            # print(pred)
            batch_size = list(pred.size())[1]
            pred_viz = pred.detach().cpu()
            pred_max = torch.argmax(pred_viz, dim=2)
            # print(f"pred max {pred_max.size()}")
            # print(pred_max)
            flat_pred = torch.zeros(pred_viz.size(), device='cpu')
            for i in range(256):
                for j in range(batch_size):
                    # print(argmax[i])
                    flat_pred[i, j, pred_max[i, j]] = 1
            # print(f"flat-pred: {flat_pred}")
            for example in range(num_plots):
                plot_pred_and_target(flat_pred[:,example,:].detach().numpy(),
                                     batch[:,example,:].detach().numpy(), is_eval)
        model.train()

    def compute_loss(self, step, model, batch, use_teacher_forcing=True, da=None):
        batch.to(device)
        pred, mu, sigma, z = model(batch, use_teacher_forcing, da)

        # Old ELBO's
        #elbo, kl = ELBO(pred, batch, mu, sigma, self.free_bits)
        # newer ELBO
        r_loss, kl_cost, kl_div, ham_dist, acc = custom_ELBO(pred, batch, mu, sigma, self.free_bits)
        kl_weight = self.KL_annealing(step, 0, 0.2)
        elbo = r_loss + kl_weight*kl_cost

        # print(f"Scores for batch: {step}")
        # print(f"R_loss: {r_loss}")
        # print(f"Elbo: {elbo}")
        # print(f"KL weight: {kl_weight}")
        # print(f"Hamming distance: {ham_dist}")
        # print(f"Batch mean KL Div: {kl_div.mean()}")

        wandb.log({"KL Weight": kl_weight, "Pred": wandb.Histogram(pred.cpu().detach().numpy())})
        # print()
        # print(sigma)
        #return kl_weight*elbo, kl
        wandb.log({"Z": wandb.Histogram(z.cpu().detach().numpy()), "mu": wandb.Histogram(mu.cpu().detach().numpy()), "sigma": wandb.Histogram(sigma.cpu().detach().numpy())})
        return elbo, kl_div.mean(), r_loss, acc, ham_dist

    def compute_flat_loss(self, step, model, batch, use_teacher_forcing=True, da=None):
        batch.to(device)
        pred, mu, sigma, z = model(batch, use_teacher_forcing, da)

        r_loss, kl_cost, kl_div = flat_ELBO(pred, batch, mu, sigma, self.free_bits)
        kl_weight = self.KL_annealing(step, 0, 0.2)
        elbo = r_loss + kl_weight * kl_cost

        acc = 0.
        ham_dist = 0.
        wandb.log({"Z": wandb.Histogram(z.cpu().detach().numpy()), "mu": wandb.Histogram(mu.cpu().detach().numpy()),
                   "sigma": wandb.Histogram(sigma.cpu().detach().numpy())})
        return elbo, r_loss, kl_div, acc, ham_dist

    def r_loss_only(self, step, model, batch, use_teacher_forcing=True, da=None):
        batch.to(device)
        pred, mu, sigma, z = model(batch, use_teacher_forcing, da)

        r_loss = only_r_loss(pred, batch, mu, sigma, self.free_bits)
        loss = Variable(r_loss, requires_grad=True)
        acc = 0.
        ham_dist = 0.
        kl_div = 0.
        wandb.log({"Z": wandb.Histogram(z.cpu().detach().numpy()), "mu": wandb.Histogram(mu.cpu().detach().numpy()),
                   "sigma": wandb.Histogram(sigma.cpu().detach().numpy())})
        return loss, r_loss, kl_div, acc, ham_dist

    def train_batch(self, iter, model, batch, da=None):
        self.optimizer.zero_grad()
        use_teacher_forcing = self.inverse_sigmoid(iter)
        #elbo, kl, r_loss, acc, ham_dist = self.compute_loss(iter, model, batch, use_teacher_forcing, da)
        #elbo, r_loss, kl_div, acc, ham_dist = self.compute_flat_loss(iter, model, batch, use_teacher_forcing, da)
        elbo, r_loss, kl_div, acc, ham_dist = self.r_loss_only(iter, model, batch, use_teacher_forcing, da)
        #print(f"elbo train batch: {elbo}")
        elbo.backward()
        self.optimizer.step()
        self.scheduler.step()
        #print(f"elbo train batch: {elbo}")

        # send batch loss data to wandb - regular loss functions
        if kl_div != 0:
            wandb.log({ "Iteration": iter, "train ELBO (batch avg)": elbo.item(), "train KL Div": kl_div.cpu(),
                    "LR": self.scheduler.get_last_lr() }) #, "Hamming Dist": ham_dist})
        else:
            # send batch loss data to wandb - r_loss only loss function
            wandb.log({"Iteration": iter, "train ELBO (batch avg)": elbo.item(), "LR": self.scheduler.get_last_lr()})  # , "Hamming Dist": ham_dist})

        # log additional metrics
        wandb.log({ "training R_loss": r_loss })#, "Training Accuracy": acc})

        if kl_div != 0:
            return elbo.item(), torch.mean(kl_div)
        else:
            return elbo.item(), 0.
        
    def train_epochs(self, model, start_epoch, iter, end_epoch, train_data, val_data=None):
        train_loss, train_kl = [], []
        val_elbo,  val_kl, val_r_loss, val_acc, val_ham_dist = [], [], [], [], []
        use_da = self.use_danceability
        print("\n   ---   Starting training   ---")
        if use_da:
            print("Training MusicVAE model WITH Danceability as additional feature ")
        else:
            print('Training regular MusicVAE')

        print(f"Training batches: {len(train_data)}")
        print(f"Val data batches: {len(val_data)}")
        print()

        for epoch in range(start_epoch, end_epoch):
            # save the randomly initialized model right away
            # self.save_checkpoint(model, epoch, iter)
            batch_loss, batch_kl = [], []
            with tqdm(total=len(train_data)) as t:
                for idx, batch in enumerate(train_data):
                    model.train()
                    # first, get data AND danceability from the dataset
                    data, da = batch
                    data = data.transpose(0, 1).squeeze()
                    data = data.to(device)
                    if use_da:
                        da = da.to(device)
                        # pass it both to the trainer
                        elbo, kl = self.train_batch(iter, model, data, da)
                    else:
                        elbo, kl = self.train_batch(iter, model, data, da=None)
                    #print(f"trainer elbo: {elbo}")
                    #print(f"batch_loss: {batch_loss}")
                    batch_loss.append(elbo)
                    batch_kl.append(kl)
                    iter += 1


                    if iter%self.print_every == 0:
                        loss_avg = torch.mean(torch.tensor(batch_loss))
                        div = torch.mean(torch.tensor(batch_kl))
                        print('\n\n\n\nEpoch: %d, iteration: %d, Average loss: %.4f, KL Divergence: %.4f' % (epoch, iter, loss_avg, div))


                    if iter%self.checkpoint_every == 0:
                        self.save_checkpoint(model, epoch, iter)

                    # plot the pred and targets as pianoroll
                    if iter%self.plot_every == 0:
                        if use_da:
                            self.plot_last_batch(model, data, use_teacher_forcing=False, da=da, num_plots=2, is_eval=False)
                        else:
                            self.plot_last_batch(model, data, use_teacher_forcing=False, da=None, num_plots=2, is_eval=False)

                    # tqdm
                    t.set_postfix(loss=f"{loss_avg}")
                    t.update()

            train_loss.append(torch.mean(torch.tensor(batch_loss)))
            train_kl.append(torch.mean(torch.tensor(batch_kl)))
            
            #self.save_checkpoint(model, epoch, iter)


            print("\n\n\n ---- Validation starting...")

            if val_data is not None:
                batch_elbo, batch_kl, batch_r_loss, batch_acc, batch_ham_dist = [], [], [], [], []
                with torch.no_grad():
                    model.eval()
                    with tqdm(total=len(val_data)) as t:
                        for idx, batch in enumerate(val_data):
                            data, da = batch
                            data = data.to(device)
                            data = data.transpose(0, 1).squeeze()
                            if use_da:
                                da = da.to(device)
                                elbo, kl, r_loss, acc, ham_dist = self.r_loss_only(iter, model, data, False, da)
                            else:
                                elbo, kl, r_loss, acc, ham_dist = self.r_loss_only(iter, model, data, False, da=None)
                            batch_elbo.append(elbo)
                            batch_kl.append(kl)
                            batch_r_loss.append(r_loss)
                            batch_acc.append(acc)
                            batch_ham_dist.append(ham_dist)
                            # tqdm
                            #t.set_postfix(loss=f"Elbo: {elbo}")
                            t.update()

                            # plot the pred and targets as pianoroll
                            if idx == len(val_data) - 1:
                                if use_da:
                                    self.plot_last_batch(model, data, use_teacher_forcing=False, da=da)
                                else:
                                    self.plot_last_batch(model, data, use_teacher_forcing=False, da=None)

                        print("Batch R loss")
                        print(batch_r_loss)
                        print("batch ELBO")
                        print(batch_elbo)
                        print(batch_loss)
                        print(batch_kl)
                        print(batch_acc)

                        # get avg values for validation dataset
                        val_elbo.append(torch.mean(torch.tensor(batch_loss)))
                        val_kl.append(torch.mean(torch.tensor(batch_kl)))
                        val_r_loss.append(torch.mean(torch.tensor(batch_r_loss)))
                        val_acc.append(torch.mean(torch.tensor(batch_acc)))
                        val_ham_dist.append(torch.mean(torch.tensor(batch_ham_dist)))

                    val_elbo_avg = torch.mean(torch.tensor(val_elbo))
                    div = torch.mean(torch.tensor(val_kl))
                    eval_r_loss = torch.mean(torch.tensor(val_r_loss))
                    eval_acc = torch.mean(torch.tensor(val_acc))
                    eval_ham_dist = torch.mean(torch.tensor(val_ham_dist))
                    print('----------Validation')
                    print('Epoch: %d, iteration: %d, Average loss: %.4f, KL Divergence: %.4f' % (epoch, iter, val_elbo_avg, div))
                    # send batch loss data to wandb
                    wandb.log({"Epoch": epoch, "Eval ELBO": val_elbo_avg, "Eval KL Div": div})
                    wandb.log({"Epoch": epoch, "Eval R_loss": eval_r_loss, "Eval Accuracy": eval_acc,"Eval Hamming Dist": eval_ham_dist})



        print("Final results:")
        print(train_loss)
        print(train_kl)
        print()
        print(val_elbo)
        print(val_kl)

        # send batch loss data to wandb
        #wandb.log({"Final training ELBOs": train_loss, "final train KL": train_kl, "Final val ELBOs": val_loss, "Final val KL": val_kl})

        # torch.save(open('outputs/train_loss_musicvae_batch', 'wb'), torch.tensor(train_loss))
        # torch.save(open('outputs/val_loss_musicvae_batch', 'wb'), torch.tensor(val_loss))
        # torch.save(open('outputs/train_kl_musicvae_batch', 'wb'), torch.tensor(train_kl))
        # torch.save(open('outputs/val_kl_musicvae_batch', 'wb'), torch.tensor(val_kl))

        #torch.save(torch.tensor(train_loss), open('scratch/outputs/train_loss_musicvae_batch', 'wb'))
        #torch.save(torch.tensor(val_loss), open('scratch/outputs/val_loss_musicvae_batch', 'wb'))
        #torch.save(torch.tensor(train_kl), open('scratch/outputs/train_kl_musicvae_batch', 'wb'))
        #torch.save(torch.tensor(val_kl), open('scratch/outputs/val_kl_musicvae_batch', 'wb'))
        
    def save_checkpoint(self, model, epoch, iter):
        print('Saving checkpoint')
        Checkpoint(model=model,
                    epoch=epoch,
                    step=iter,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    samp_rate=self.sampling_rate,
                    KL_rate=self.KL_rate,
                    free_bits=self.free_bits).save(self.output_dir)
        print('Checkpoint Successful')
        
    def load_checkpoint(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.output_dir)
        resume_checkpoint  = Checkpoint.load(latest_checkpoint_path)
        model              = resume_checkpoint.model
        epoch              = resume_checkpoint.epoch
        iter               = resume_checkpoint.step
        self.scheduler     = resume_checkpoint.scheduler
        self.optimizer     = resume_checkpoint.optimizer
        self.sampling_rate = resume_checkpoint.samp_rate
        self.KL_rate       = resume_checkpoint.KL_rate
        self.free_bits     = resume_checkpoint.free_bits
        return model, epoch, iter
    
    def train(self, model, train_data, optimizer, epochs, resume=False, val_data=None):
        if resume:
            model, epoch, iter = self.load_checkpoint()
        else:
            if optimizer is None:
                self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
                self.scheduler = LambdaLR(self.optimizer, decay_old)
                
            epoch = 1
            iter = 0
            print(model)
            print(self.optimizer)
            print(self.scheduler)
            print('Starting epoch %d' % epoch)

        model.to(device)
        self.train_epochs(model, epoch, iter, epoch+epochs, train_data, val_data)
                
                        
                        
            
            
            
            
                
                
