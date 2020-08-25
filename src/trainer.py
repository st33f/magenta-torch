from src.checkpoint import Checkpoint
from src.loss import ELBO, custom_ELBO

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from math import exp
import numpy as np

from tqdm import tqdm
import wandb
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def decay_old(x):
    return 0.01 + (0.99)*(0.9999)**x

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
    
    def compute_loss(self, step, model, batch, use_teacher_forcing=True, da=None):
        batch.to(device)
        pred, mu, sigma, z = model(batch, use_teacher_forcing, da)
        #elbo, kl = ELBO(pred, batch, mu, sigma, self.free_bits)
        r_loss, kl_cost, kl_div, ham_dist, acc = custom_ELBO(pred, batch, mu, sigma, self.free_bits)
        kl_weight = self.KL_annealing(step, 0, 0.2)
        elbo = r_loss + kl_weight*kl_cost
        wandb.log({"KL Weight": kl_weight, "R_loss": r_loss, "Accuracy": acc})


        pred_viz = pred.cpu()
        wandb.log({"Pred": wandb.Histogram(pred_viz.detach().numpy())})
        # print()
        print(f"R_loss: {r_loss}")
        print(f"Elbo: {elbo}")
        print(f"KL weight: {kl_weight}")
        print(f"Hamming distance: {ham_dist}")
        # print(f"Batch mean KL Div: {kl_div.mean()}")
        # print()
        #return kl_weight*elbo, kl
        return elbo, kl_div.mean()
        
    def train_batch(self, iter, model, batch, da=None):
        self.optimizer.zero_grad()
        use_teacher_forcing = self.inverse_sigmoid(iter)
        elbo, kl = self.compute_loss(iter, model, batch, use_teacher_forcing, da)
        #print(f"elbo train batch: {elbo}")
        elbo.backward()
        self.optimizer.step()
        self.scheduler.step()
        #print(f"elbo train batch: {elbo}")
        return elbo.item(), kl.item()
        
    def train_epochs(self, model, start_epoch, iter, end_epoch, train_data, val_data=None):
        train_loss, val_loss = [], []
        train_kl,  val_kl = [], []
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
            model.train()
            with tqdm(total=len(train_data)) as t:
                for idx, batch in enumerate(train_data):
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
                        # send batch loss data to wandb
                        wandb.log({"train ELBO (batch avg)": loss_avg, "train KL Div": div, "Epoch": epoch, "Iteration": iter, "LR": self.scheduler.get_last_lr()})

                    if iter%self.checkpoint_every == 0:
                        self.save_checkpoint(model, epoch, iter)

                    # tqdm
                    t.set_postfix(loss=f"{loss_avg}")
                    t.update()

            train_loss.append(torch.mean(torch.tensor(batch_loss)))
            train_kl.append(torch.mean(torch.tensor(batch_kl)))
            
            #self.save_checkpoint(model, epoch, iter)


            print("\n\n\n ---- Validation starting...")

            if val_data is not None:
                batch_loss, batch_kl = [], []
                with torch.no_grad():
                    model.eval()
                    with tqdm(total=len(val_data)) as t:
                        for idx, batch in enumerate(val_data):
                            data, da = batch
                            data = data.to(device)
                            data = data.transpose(0, 1).squeeze()
                            if use_da:
                                da = da.to(device)
                                elbo, kl = self.compute_loss(iter, model, data, False, da)
                            else:
                                elbo, kl = self.compute_loss(iter, model, data, False, da=None)
                            batch_loss.append(elbo)
                            batch_kl.append(kl)
                            # tqdm
                            #t.set_postfix(loss=f"Elbo: {elbo}")
                            t.update()
                        val_loss.append(torch.mean(torch.tensor(batch_loss)))
                        val_kl.append(torch.mean(torch.tensor(batch_kl)))
                    val_loss_avg = torch.mean(torch.tensor(val_loss))
                    div = torch.mean(torch.tensor(val_kl))
                    print('----------Validation')
                    print('Epoch: %d, iteration: %d, Average loss: %.4f, KL Divergence: %.4f' % (epoch, iter, loss_avg, div))
                    # send batch loss data to wandb
                    wandb.log({"Epoch": epoch, "Validation ELBO": val_loss_avg, "Validation KL Div": div})

        print("Final results:")
        print(train_loss)
        print(train_kl)
        print()
        print(val_loss)
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
                
                        
                        
            
            
            
            
                
                
