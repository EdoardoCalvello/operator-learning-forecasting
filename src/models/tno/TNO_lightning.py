import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt

from .TNO_pytorch import SimpleEncoder

# Define the pytorch lightning module for training the Simple Encoder model
class SimpleEncoderModule(pl.LightningModule):
    def __init__(self, input_dim=1, output_dim=1, d_model=32, nhead=8, num_layers=6,
                 domain_dim=1,
                 plot_inds=None,
                 dyn_sys_name='Lorenz63',
                 output_inds=None,
                 state_dim=128,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 use_positional_encoding='continuous',
                 append_position_to_x=False,
                 pos_enc_coeff=2,
                 loss_name = 'l2_rel',
                 activation='relu',
                 monitor_metric='loss/val/mse',
                 lr_scheduler_params={'patience': 3,
                                      'factor': 0.5},
                 dropout=0.1,
                 norm_first=False,
                 dim_feedforward=2048,
                 query_seq_len=None,    # int(prediction_horizon / dt); enables cross-attention decoder
                 query_coords=None,     # (query_seq_len, domain_dim) tensor of query time coords from config
                 output_seq_len=None):  # encoder-only forecast output length
        super(SimpleEncoderModule, self).__init__()
        self.first_forward = True
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.use_positional_encoding = use_positional_encoding
        self.append_position_to_x = append_position_to_x
        self.monitor_metric = monitor_metric
        self.lr_scheduler_params = lr_scheduler_params
        self.domain_dim = domain_dim
        self.output_dim = output_dim
        self.loss_name = loss_name

        self.model = SimpleEncoder(input_dim=input_dim,
                                    output_dim=output_dim,
                                    domain_dim=domain_dim,
                                    d_model=d_model, 
                                    nhead=nhead, 
                                    num_layers=num_layers,
                                    max_sequence_length=max_sequence_length,
                                    do_layer_norm=do_layer_norm,
                                    use_transformer=use_transformer,
                                    use_positional_encoding=use_positional_encoding,
                                    append_position_to_x=append_position_to_x,
                                    pos_enc_coeff=pos_enc_coeff,
                                    activation=activation,
                                    dropout=dropout,
                                    norm_first=norm_first,
                                    dim_feedforward=dim_feedforward,
                                    query_seq_len=query_seq_len,
                                    query_coords=query_coords,
                                    output_seq_len=output_seq_len)
        
        self.test_losses = {}
        self.plot_inds = plot_inds or range(output_dim)  # Default to all dimensions if not specified
        self.dyn_sys_name = dyn_sys_name
        self.output_inds = output_inds or list(range(output_dim))
        self.state_dim = state_dim


    def forward(self, x, y, coords_x, coords_y):
        coords_x = coords_x[0].unsqueeze(2)
        # coords_y: (batch, seq_len_y, domain_dim) → take one since all are the same
        coords_y = coords_y[0]  # (seq_len_y, domain_dim)
        if self.first_forward:
            self.first_forward = False

        return self.model(x, y=None, coords_x=coords_x, coords_y=coords_y)

    def training_step(self, batch, batch_idx):
        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/train/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("loss/train/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss
        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log("loss/train/rel_L2", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)


        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, coords_x, coords_y, tag='Train')

        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]

    def on_after_backward(self):
        self.log_gradient_norms(tag='afterBackward')

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer and its gradient
        # If using mixed precision, the gradients are already unscaled here
        self.log_gradient_norms(tag='beforeOptimizer')
        self.log_parameter_norms(tag='beforeOptimizer')

    def log_gradient_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(norm_type)
                name = name.replace('.', '_')
                self.log(f"grad_norm/{tag}/{name}", grad_norm,
                         on_step=False, on_epoch=True, prog_bar=False)

    def log_parameter_norms(self, tag=''):
        norm_type = 2.0
        for name, param in self.named_parameters():
            param_norm = param.detach().norm(norm_type)
            name = name.replace('.', '_')
            self.log(f"param_norm/{tag}/{name}", param_norm,
                     on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log("loss/val/mse", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log("loss/val/sup", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss

        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log("loss/val/rel_L2", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self.make_batch_figs(x, y, y_hat, coords_x, coords_y, tag='Val')

        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]

    def plot_positional_encoding(self, x, coords):
        pe = self.model.positional_encoding(x, coords)
        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.imshow(pe.detach().cpu().clone().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Positional Encoding')
        wandb.log({f"plots/Positional Encoding": wandb.Image(fig)})
        plt.close()

    def make_batch_figs(self, x, y, y_hat, coords_x, coords_y, tag='', n_examples=4):
        if n_examples > x.shape[0]:
            n_examples = x.shape[0]
            
        # Select representative trajectories instead of random ones
        try:
            selected_indices, labels = self.select_representative_trajectories(
                y, y_hat, n_total=min(n_examples, x.shape[0])
            )
            print(f"🔍 Debug: batch_size={x.shape[0]}, selected_indices={selected_indices.tolist()}, labels={labels}")
            idx = selected_indices
        except Exception as e:
            print(f"⚠️  Error in trajectory selection: {e}")
            # Fallback to simple indices
            idx = torch.arange(min(n_examples, x.shape[0]))
            labels = [f"Traj_{i}" for i in range(len(idx))]
        
        # CRITICAL FIX: Denormalize data before plotting for physical interpretation
        y_pred = y_hat[idx].detach().cpu()
        y_true = y[idx].detach().cpu()
        
        # FIXED: Denormalize using normalizers passed from experiment
        if hasattr(self, 'y_normalizer') and self.y_normalizer is not None:
            try:
                y_pred = self.y_normalizer.decode(y_pred)
                y_true = self.y_normalizer.decode(y_true)
                print(f"📊 DENORMALIZED plots for {tag}: Physical values restored!")
            except Exception as e:
                print(f"⚠️  WARNING: Could not denormalize {tag} plots: {e}")
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            # Fallback to datamodule if available
            try:
                if hasattr(self.trainer.datamodule.dataset, 'y_normalizer'):
                    y_normalizer = self.trainer.datamodule.dataset.y_normalizer
                    y_pred = y_normalizer.decode(y_pred)
                    y_true = y_normalizer.decode(y_true)
                    print(f"📊 DENORMALIZED plots for {tag}: Physical values restored!")
                else:
                    print(f"⚠️  WARNING: No y_normalizer found - plotting normalized values for {tag}")
            except Exception as e:
                print(f"⚠️  WARNING: Could not denormalize {tag} plots: {e}")
        else:
            print(f"⚠️  WARNING: No normalizers available - plotting normalized values for {tag}")

        # Convert to numpy for plotting
        y_pred = y_pred.clone().numpy()
        y_true = y_true.clone().numpy()
        coords_x = coords_x[idx].detach().cpu().clone().numpy()
        coords_y = coords_y[idx].detach().cpu().clone().numpy()

        # Use sequential indices for the extracted trajectories (0, 1, 2, ...)
        sequential_idx = torch.arange(len(idx))

        if self.domain_dim == 1:
            self.batch_figs_1D_representative(x, y_true, y_pred, coords_x, coords_y, tag, sequential_idx, labels)
        elif self.domain_dim == 2:
            self.batch_figs_2D_representative(x, y_true, y_pred, coords_x, coords_y, tag, sequential_idx, labels)

    def make_test_figs(self, median_sample, worst_sample, tag=None):

        if self.domain_dim == 1:
            self.test_figs_1D(median_sample, worst_sample, tag)
        elif self.domain_dim == 2:
            self.test_figs_2D(median_sample, worst_sample, tag)

    def batch_figs_1D(self, x, y_true, y_pred, coords_x, coords_y, tag, idx):
        # Get plot indices from dataset config
        plot_inds = self.plot_inds
        
        # Plot Trajectories
        plt.figure()
        fig, axs = plt.subplots(
            nrows=len(plot_inds), 
            ncols=len(idx), 
            figsize=(10 * len(idx), 6 * len(plot_inds)), 
            sharex=True, 
            squeeze=False
        )

        for col, idx_val in enumerate(idx):
            for i, plot_idx in enumerate(plot_inds):
                ax = axs[i, col]
                ax.plot(coords_y[idx_val], y_true[idx_val, :, plot_idx], 
                        linewidth=3, color='blue', label='Ground Truth')
                ax.plot(coords_y[idx_val], y_pred[idx_val, :, plot_idx], 
                        linewidth=3, color='red', label='Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'Prediction (dim {plot_idx})')
                ax.set_title(f'Trajectory for predicted component {plot_idx} (Index {idx_val})')
                if col == 0:
                    ax.legend()

        fig.suptitle(f'{tag} Trajectories: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        wandb.log({f"plots/{tag}/Trajectories: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()
        plt.close('all')

    def batch_figs_2D(self, x, y_true, y_pred, coords_x, coords_y, tag, idx, n_grid=100):

        # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
        # plot the values of y_true and y_pred at the indices given by coords_y

        # Plot a 3 paneled figure with 3 scalar 2-d fields (heatmaps)
        # 1. Ground truth
        # 2. Prediction
        # 3. Relative difference

        # get the low and high indices of the y coordinates
        i_low_1, i_low_2 = np.min(coords_y[...,0]), np.min(coords_y[...,1])
        i_high_1, i_high_2 = np.max(coords_y[...,0]), np.max(coords_y[...,1])
        # build a meshgrid of coordinates based on coords_y
        y1i, y2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        # get the low and high indices of the x coordinates
        i_low_1, i_low_2 = np.min(coords_x[...,0]), np.min(coords_x[...,1])
        i_high_1, i_high_2 = np.max(coords_x[...,0]), np.max(coords_x[...,1])
        # build a meshgrid of coordinates based on coords_y
        x1i, x2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=len(idx), figsize=(10 * len(idx), 6 * 4), sharex=True, squeeze=False)

        for col, idx_val in enumerate(idx):
            x_input_i = griddata(
                (coords_x[idx_val, :, 0], coords_x[idx_val, :, 1]), x[idx_val].detach().cpu().numpy(), (x1i, x2i), method='linear')
            y_true_i = griddata(
                (coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_true[idx_val], (y1i, y2i), method='linear')
            y_pred_i = griddata((coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_pred[idx_val], (y1i, y2i), method='linear')
            #y_true_i_norm = np.sqrt((1/(y_true_i.shape[0]*y_true_i.shape[1]))*np.sum(y_true_i**2))
            y_rel_diff_i = np.abs(y_pred_i - y_true_i) #/ np.abs(y_true_i + 1e-5)
            #y_rel_diff_i = np.abs(y_pred_i - y_true_i)

            #plot median and worst case relative error

            for i, ax in enumerate(axs[:, col]):
                if i == 0:
                    # plot input field x
                    im = ax.imshow(x_input_i, cmap='viridis')
                    ax.set_title(
                        f'Input Field (Index {idx_val})')

                if i == 1:
                    im = ax.imshow(y_true_i, cmap='viridis')
                    ax.set_title(
                        f'Ground Truth (Index {idx_val})')
                elif i == 2:
                    im = ax.imshow(y_pred_i, cmap='viridis')
                    ax.set_title(
                        f'Prediction (Index {idx_val})')
                elif i == 3:
                    # plot absolute relative error in log scale (difference divided by ground truth)
                    #im = ax.imshow(np.log10(y_rel_diff_i + 1e-10), cmap='viridis', vmin=-5, vmax=3)
                    im = ax.imshow(np.log10(y_rel_diff_i), cmap='inferno', vmin=-7,vmax=1)
                    ax.set_title(
                        f'Pointwise Error (Index {idx_val})')
                fig.colorbar(im, ax=ax)

        fig.suptitle(f'{tag} Predicted Fields: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
            {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Use a default sample rate if datamodule not available
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]
        else:
            dt = 0.01  # Default sample rate
            
        x, y, coords_x, coords_y = batch
        y_hat = self.forward(x, y, coords_x, coords_y)
        loss = F.mse_loss(y_hat, y)
        self.log(f"loss/test/mse/dt{dt}", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Sup norm loss
        loss_sup  = torch.max(torch.abs(y_hat - y))
        self.log(f"loss/test/sup/dt{dt}", loss_sup, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        #Relative L2 loss

        rel_loss = torch.mean(torch.div(torch.sqrt(torch.mean(torch.mean((y_hat -y)**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y**2, dim=1),dim=1))))
        self.log(f"loss/test/rel_L2/dt{dt}", rel_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        
        # Save the test losses and corresponding indices for each DataLoader index
        if dataloader_idx not in self.test_losses:
            self.test_losses[dataloader_idx] = {'losses': [], 'indices': [], 'prediction':[]}
        
        for i in range(x.size(0)):
            loss_sample = torch.div(torch.sqrt(torch.mean(torch.mean((y_hat[i:i+1] -y[i:i+1])**2, dim=1),dim=1)),
                                        torch.sqrt(torch.mean(torch.mean(y[i:i+1]**2, dim=1),dim=1)))

            # Save the test losses and corresponding indices for each DataLoader index
            self.test_losses[dataloader_idx]['losses'].append(loss_sample.cpu().detach().clone().numpy())
            self.test_losses[dataloader_idx]['indices'].append((batch_idx, i))
            self.test_losses[dataloader_idx]['prediction'].append(y_hat[i:i+1])


        loss_dict = {"l2": loss, "l2_rel": rel_loss}
        return loss_dict[self.loss_name]
    

    def test_figs_1D(self, median_sample, worst_sample, tag):
        x_median, y_median, coords_x_median, coords_y_median, y_pred_median, median_error = median_sample
        x_min, y_min, coords_x_min, coords_y_min, y_pred_min, min_error = worst_sample

        # Convert to numpy arrays for plotting
        x_median = x_median.cpu().numpy()
        y_median = y_median.cpu().numpy()
        coords_x_median = coords_x_median.cpu().numpy()
        coords_y_median = coords_y_median.cpu().numpy()
        y_pred_median = y_pred_median.cpu().numpy()
        median_error = median_error[0]

        x_min = x_min.cpu().numpy()
        y_min = y_min.cpu().numpy()
        coords_x_min = coords_x_min.cpu().numpy()
        coords_y_min = coords_y_min.cpu().numpy()
        y_pred_min = y_pred_min.cpu().numpy()
        min_error = min_error[0]

        # Get plot indices from dataset config
        plot_inds = self.plot_inds
        
        plt.figure()
        fig, axs = plt.subplots(
            nrows=(len(plot_inds)+1),  # Changed from self.output_dim to len(plot_inds)
            ncols=2,
            figsize=(14 * len(plot_inds), 7 * len(plot_inds)),  # Adjusted figure size
            sharex=True,
            squeeze=False,
        )

        axs[0, 1].set_title(
                f"Trajectories for predicted components: \nMaximum Relative L2 Error ({min_error:.2e})"
            )

        axs[0, 0].set_title(
            f"Trajectories for predicted components: \nMedian Relative L2 Error ({median_error:.2e})"
        )

        # In the first column, plot the inputs

        # Min
        axs[0, 1].plot(
            coords_x_min[:, 0],
            x_min[:, 0],
            linewidth=3,
            color="blue",
            label=f"Input",
        )
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].legend()

        # Median
        axs[0, 0].plot(
            coords_x_median[:, 0],
            x_median[:, 0],
            linewidth=3,
            color="blue",
            label=f"Input",
        )
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].legend()

        for i, plot_idx in enumerate(plot_inds):
            # Min case plotting
            axs[i+1, 1].plot(
                coords_y_min[:, 0],
                y_min[:, plot_idx],
                linewidth=3,
                color="blue",
                label="Ground Truth",
            )
            axs[i+1, 1].plot(
                coords_y_min[:, 0],
                y_pred_min[0, :, plot_idx],
                linewidth=3,
                color="red",
                label="Prediction",
            )
            axs[i+1, 1].set_xlabel("Time")
            axs[i+1, 1].set_ylabel(f"Prediction (dim {plot_idx})")
            axs[i+1, 1].legend()

            # Median case plotting
            axs[i+1, 0].plot(
                coords_y_median[:, 0],
                y_median[:, plot_idx],
                linewidth=3,
                color="blue",
                label="Ground Truth",
            )
            axs[i+1, 0].plot(
                coords_y_median[:, 0],
                y_pred_median[0, :, plot_idx],
                linewidth=3,
                color="red",
                label="Prediction",
            )
            axs[i+1, 0].set_xlabel("Time")
            axs[i+1, 0].set_ylabel(f"Prediction (dim {plot_idx})")
            axs[i+1, 0].legend()

            # If you want to add space between rows, you can adjust the hspace parameter here.

        fig.suptitle(f"Trajectories: Prediction vs. Truth (sample rate = {tag})")
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
                {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()


    def test_figs_2D(self, median_sample, worst_sample, tag):


        x_median, y_median, coords_x_median, coords_y_median,y_pred_median, median_error = median_sample
        x_min, y_min, coords_x_min, coords_y_min, y_pred_min, min_error = worst_sample

        #make all the previous into numpy arrays
        x_median, y_median, coords_x_median, coords_y_median, y_pred_median, median_error = x_median.cpu().clone().numpy(), y_median.cpu().clone().numpy(), coords_x_median.cpu().clone().numpy(), coords_y_median.cpu().clone().numpy(), y_pred_median.cpu().clone().numpy(), median_error[0]
        x_min, y_min, coords_x_min, coords_y_min , y_pred_min, min_error = x_min.cpu().clone().numpy(), y_min.cpu().clone().numpy(), coords_x_min.cpu().clone().numpy(), coords_y_min.cpu().clone().numpy() , y_pred_min.cpu().clone().numpy(), min_error[0]

        # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
        # plot the values of y_true and y_pred at the indices given by coords_y
        n_grid=100

        # get the low and high indices of the y coordinates
        i_low_1, i_low_2 = np.min(coords_y_min[...,0]), np.min(coords_y_min[...,1])
        i_high_1, i_high_2 = np.max(coords_y_min[...,0]), np.max(coords_y_min[...,1])
        # build a meshgrid of coordinates based on coords_y
        y1i_min, y2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        # get the low and high indices of the x coordinates
        i_low_1, i_low_2 = np.min(coords_x_min[...,0]), np.min(coords_x_min[...,1])
        i_high_1, i_high_2 = np.max(coords_x_min[...,0]), np.max(coords_x_min[...,1])
        # build a meshgrid of coordinates based on coords_y
        x1i_min, x2i_min = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        # get the low and high indices of the y coordinates
        i_low_1_median, i_low_2_median = np.min(coords_y_median[...,0]), np.min(coords_y_median[...,1])
        i_high_1_median, i_high_2_median = np.max(coords_y_median[...,0]), np.max(coords_y_median[...,1])
        # build a meshgrid of coordinates based on coords_y
        y1i_median, y2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median, i_high_2_median, n_grid))

        # get the low and high indices of the x coordinates
        i_low_1_median, i_low_2_median = np.min(coords_x_median[...,0]), np.min(coords_x_median[...,1])
        i_high_1_median, i_high_2_median = np.max(coords_x_median[...,0]), np.max(coords_x_median[...,1])
        # build a meshgrid of coordinates based on coords_y
        x1i_median, x2i_median = np.meshgrid(np.linspace(i_low_1_median, i_high_1_median, n_grid), np.linspace(i_low_2_median, i_high_2_median, n_grid))

        plt.figure()
        fig, axs = plt.subplots(
            nrows=2, ncols=4, figsize=(24, 12), sharex=True, squeeze=False)

        x_input_min = griddata(
                (coords_x_min[:, 0], coords_x_min[:, 1]), x_min, (x1i_min, x2i_min), method='linear')
        y_true_min = griddata(
                (coords_y_min[ :, 0], coords_y_min[ :, 1]), y_min, (y1i_min, y2i_min), method='linear')
        y_pred_min = griddata((coords_y_min[ :, 0], coords_y_min[ :, 1]), y_pred_min[0,:,:], (y1i_min, y2i_min), method='linear')
        y_rel_diff_min = np.abs(y_pred_min - y_true_min)

        x_input_median = griddata(
                (coords_x_median[:, 0], coords_x_median[:, 1]), x_median, (x1i_median, x2i_median), method='linear')
        y_true_median = griddata(
                (coords_y_median[:, 0], coords_y_median[:, 1]), y_median, (y1i_median, y2i_median), method='linear')
        y_pred_median = griddata((coords_y_median[ :, 0], coords_y_median[ :, 1]), y_pred_median[0,:,:], (y1i_median, y2i_median), method='linear')
        y_rel_diff_median = np.abs(y_pred_median - y_true_median)

        data_sets = {
            (0, 0): x_input_median,
            (0, 1): y_true_median,
            (0, 2): y_pred_median,
            (0, 3): np.log10(y_rel_diff_median),
            (1, 0): x_input_min,
            (1, 1): y_true_min,
            (1, 2): y_pred_min,
            (1, 3): np.log10(y_rel_diff_min)
        }

        for (i, j), data in data_sets.items():
            bounds = (-7, 1) if j == 3 else (None, None)
            im = axs[i, j].imshow(data, cmap='viridis' if j < 3 else 'inferno', vmin=bounds[0], vmax=bounds[1])
            error_info = f"Median Relative L2 Error ({median_error:.2e})" if i == 0 else f"Maximum Relative L2 Error ({min_error:.2e})"
            axs[i, j].set_title(
                f"{'Input Field: ' if j == 0 else 'Ground Truth:' if j == 1 else 'Prediction:' if j == 2 else 'Pointwise Error:'}\n "
                f"{error_info}", 
                fontsize=14
            )

            fig.colorbar(im, ax=axs[i, j], shrink=0.6)


        fig.suptitle(f'{tag} Predicted Fields: Prediction vs. Truth', fontsize=30)
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        wandb.log(
                {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()
    
    def on_test_epoch_end(self):

        for dataloader_idx, dataloader_losses in self.test_losses.items():
            losses = torch.tensor(dataloader_losses['losses'])

            # Calculate median and worst error indices
            median_idx = int(torch.argsort(losses,dim=0)[len(losses) // 2])
            worst_idx = int(torch.argmax(losses))

            # Access indices of median and worst cases
            median_batch_idx, median_sample_idx = dataloader_losses['indices'][median_idx]
            worst_batch_idx, worst_sample_idx = dataloader_losses['indices'][worst_idx]

            # Access the test data loaders
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                dt = self.trainer.datamodule.test_sample_rates[dataloader_idx]
                test_dataloader = self.trainer.datamodule.test_dataloader()[dt]
            else:
                dt = 0.01  # Default sample rate
                # Skip plotting if no datamodule available
                continue

            # Retrieve the samples using indices, 
            ###################################################
            ###################################################
            # !!!!!terrible code!!!!!
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx == median_batch_idx:
                    median_batch = batch
                if batch_idx == worst_batch_idx:
                    worst_batch = batch


            median_sample = [median_batch[i][median_sample_idx] for i in range(4)]+[dataloader_losses['prediction'][median_idx], dataloader_losses['losses'][median_idx]]
            worst_sample =  [worst_batch[i][worst_sample_idx] for i in range(4)]+[dataloader_losses['prediction'][worst_idx], dataloader_losses['losses'][worst_idx]]
            ###################################################
            ###################################################

            # Now you can use median_sample and worst_sample to plot relevant information or perform further analysis
    
            self.make_test_figs(median_sample, worst_sample, tag=f'Test/dt{dt}')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        config = {
            # REQUIRED: The scheduler instance
            "scheduler": ReduceLROnPlateau(optimizer, **self.lr_scheduler_params),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.monitor_metric,  # "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": config,
        }

    def select_representative_trajectories(self, y_true, y_pred, n_total=4):
        """
        Select representative trajectories: mean, median, worst (highest error), best (lowest error).
        
        Args:
            y_true: Ground truth trajectories (batch_size, seq_len, features)
            y_pred: Predicted trajectories (batch_size, seq_len, features)
            n_total: Number of trajectories to select (default 4)
            
        Returns:
            indices: Selected trajectory indices
            labels: Labels for each selected trajectory
        """
        # Compute error for each trajectory using the configured loss function
        if self.loss_name == 'l2_rel':
            # Relative L2 error per trajectory
            numerator = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=(1, 2)))  # (batch_size,)
            denominator = torch.sqrt(torch.mean(y_true**2, dim=(1, 2))) + 1e-8  # (batch_size,)
            error_per_traj = numerator / denominator
        else:
            # Default to MSE per trajectory
            error_per_traj = torch.mean((y_pred - y_true)**2, dim=(1, 2))  # (batch_size,)
        
        # Get indices
        n_trajs = len(error_per_traj)
        sorted_indices = torch.argsort(error_per_traj)
        
        if n_total >= n_trajs:
            # If we have fewer trajectories than requested, return all
            return torch.arange(n_trajs), [f"Traj_{i}" for i in range(n_trajs)]
        
        # Select representative trajectories
        selected_indices = []
        labels = []
        
        if n_total >= 1:
            # Best (lowest error)
            selected_indices.append(sorted_indices[0].item())
            labels.append("Best")
            
        if n_total >= 2:
            # Worst (highest error)  
            selected_indices.append(sorted_indices[-1].item())
            labels.append("Worst")
            
        if n_total >= 3:
            # Median
            median_idx = sorted_indices[len(sorted_indices) // 2].item()
            selected_indices.append(median_idx)
            labels.append("Median")
            
        if n_total >= 4:
            # Mean-like (trajectory with error closest to mean error)
            mean_error = torch.mean(error_per_traj)
            mean_like_idx = torch.argmin(torch.abs(error_per_traj - mean_error)).item()
            # Avoid duplicate if mean-like is same as best, worst, or median
            if mean_like_idx not in selected_indices:
                selected_indices.append(mean_like_idx)
                labels.append("Mean")
            else:
                # Find the next closest trajectory to mean error among remaining trajectories
                remaining_indices = [i for i in range(n_trajs) if i not in selected_indices]
                if remaining_indices:
                    # Among remaining trajectories, find the one closest to mean error
                    remaining_errors = error_per_traj[remaining_indices]
                    closest_to_mean_idx = torch.argmin(torch.abs(remaining_errors - mean_error)).item()
                    actual_idx = remaining_indices[closest_to_mean_idx]
                    selected_indices.append(actual_idx)
                    labels.append("Mean")
                else:
                    # Fallback if somehow no remaining indices (shouldn't happen)
                    selected_indices.append(mean_like_idx)
                    labels.append("Mean")
        
        # Ensure we have exactly n_total trajectories (fill with additional if needed)
        while len(selected_indices) < n_total and len(selected_indices) < n_trajs:
            # Find unused indices
            unused = [i for i in range(n_trajs) if i not in selected_indices]
            if unused:
                selected_indices.append(unused[0])
                labels.append(f"Extra_{len(selected_indices)}")
            else:
                break
        
        # Take only the requested number
        selected_indices = selected_indices[:n_total]
        labels = labels[:n_total]
        
        return torch.tensor(selected_indices), labels

    def plot_final_predictions(self, test_loader, n_examples=4):
        """Create comprehensive final plots after training completes."""
        print("🎨 Creating final prediction plots...")
        
        self.eval()
        all_inputs = []
        all_predictions = []
        all_targets = []
        all_coords_x = []
        all_coords_y = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(all_predictions) >= 50:  # Collect enough data for good selection
                    break
                x, y, coords_x, coords_y = batch
                x, y = x.to(self.device), y.to(self.device)
                coords_x, coords_y = coords_x.to(self.device), coords_y.to(self.device)
                
                y_hat = self.forward(x, y, coords_x, coords_y)
                
                # Properly collect ALL data including inputs
                all_inputs.append(x.cpu())
                all_predictions.append(y_hat.cpu())
                all_targets.append(y.cpu())
                all_coords_x.append(coords_x.cpu())
                all_coords_y.append(coords_y.cpu())
        
        if all_predictions:
            # Concatenate all batches properly
            x_all = torch.cat(all_inputs, dim=0)
            y_pred_all = torch.cat(all_predictions, dim=0)
            y_true_all = torch.cat(all_targets, dim=0)
            coords_x_all = torch.cat(all_coords_x, dim=0)
            coords_y_all = torch.cat(all_coords_y, dim=0)
            
            # Always select exactly 4 representative trajectories for consistency with Train/Val
            selected_indices, labels = self.select_representative_trajectories(
                y_true_all, y_pred_all, n_total=4  # Force 4 trajectories
            )
            
            print(f"📊 Selected trajectories: {', '.join(labels)}")
            
            # Create modified make_batch_figs that shows selected trajectories with labels
            self.make_representative_figs(
                x_all, y_true_all, y_pred_all, coords_x_all, coords_y_all, 
                selected_indices, labels, tag='Test'
            )
            print(f"✅ Final plots created with {len(selected_indices)} representative samples")
        
        self.train()  # Return to training mode

    def make_representative_figs(self, x, y_hat, y_true, coords_x, coords_y, selected_indices, labels, tag=''):
        """Create plots with representative trajectories (mean, median, worst, best) with labels."""
        idx = selected_indices
        n_examples = len(idx)
        
        # CRITICAL FIX: Denormalize data before plotting for physical interpretation
        y_pred = y_hat[idx].detach().cpu()
        y_true = y_true[idx].detach().cpu()
        
        # FIXED: Denormalize using normalizers passed from experiment
        if hasattr(self, 'y_normalizer') and self.y_normalizer is not None:
            try:
                y_pred = self.y_normalizer.decode(y_pred)
                y_true = self.y_normalizer.decode(y_true)
                print(f"📊 DENORMALIZED plots for {tag}: Physical values restored!")
            except Exception as e:
                print(f"⚠️  WARNING: Could not denormalize {tag} plots: {e}")
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            # Fallback to datamodule if available
            try:
                if hasattr(self.trainer.datamodule.dataset, 'y_normalizer'):
                    y_normalizer = self.trainer.datamodule.dataset.y_normalizer
                    y_pred = y_normalizer.decode(y_pred)
                    y_true = y_normalizer.decode(y_true)
                    print(f"📊 DENORMALIZED plots for {tag}: Physical values restored!")
                else:
                    print(f"⚠️  WARNING: No y_normalizer found - plotting normalized values for {tag}")
            except Exception as e:
                print(f"⚠️  WARNING: Could not denormalize {tag} plots: {e}")
        else:
            print(f"⚠️  WARNING: No normalizers available - plotting normalized values for {tag}")
        

        # Convert to numpy for plotting
        y_pred = y_pred.clone().numpy()
        y_true = y_true.clone().numpy()
        coords_x = coords_x[idx].detach().cpu().clone().numpy()
        coords_y = coords_y if isinstance(coords_y, np.ndarray) else coords_y.detach().cpu().clone().numpy()

        # Use sequential indices for the extracted trajectories (0, 1, 2, ...)
        sequential_idx = torch.arange(n_examples)

        if self.domain_dim == 1:
            self.batch_figs_1D_representative(x, y_true, y_pred, coords_x, coords_y, tag, sequential_idx, labels)
        elif self.domain_dim == 2:
            self.batch_figs_2D_representative(x, y_true, y_pred, coords_x, coords_y, tag, sequential_idx, labels)

    def batch_figs_1D_representative(self, x, y_true, y_pred, coords_x, coords_y, tag, idx, labels):
        """Plot 1D trajectories with descriptive labels."""
        # Use self.plot_inds instead of accessing dataset config
        plot_inds = self.plot_inds
        
        plt.figure()
        fig, axs = plt.subplots(
            nrows=len(plot_inds), 
            ncols=len(idx), 
            figsize=(10 * len(idx), 6 * len(plot_inds)),  
            sharex=True, 
            squeeze=False
        )

        for col, (idx_val, label) in enumerate(zip(idx, labels)):
            for i, plot_idx in enumerate(plot_inds):
                ax = axs[i, col]
                ax.plot(coords_y[idx_val], y_true[idx_val, :, plot_idx], 
                        linewidth=3,  
                        color='blue', 
                        label='Ground Truth')
                ax.plot(coords_y[idx_val], y_pred[idx_val, :, plot_idx], 
                        linewidth=3,  
                        color='red', 
                        label='Prediction')
                ax.set_xlabel('Time')
                ax.set_ylabel(f'Dim {plot_idx}')
                ax.set_title(f'{label} - Component {plot_idx}', fontsize=8)  # Reduced font size
                if col == 0:
                    ax.legend()

        fig.suptitle(f'{tag} Trajectories: Prediction vs. Truth (Representative Samples)')
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
            {f"plots/{tag}/Trajectories: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()
        plt.close('all')

    def batch_figs_2D_representative(self, x, y_true, y_pred, coords_x, coords_y, tag, idx, labels):

        # Each element of y_true and y_pred is a 2D field with coordinates given by coords_y
        # plot the values of y_true and y_pred at the indices given by coords_y

        # Plot a 3 paneled figure with 3 scalar 2-d fields (heatmaps)
        # 1. Ground truth
        # 2. Prediction
        # 3. Relative difference

        # get the low and high indices of the y coordinates
        i_low_1, i_low_2 = np.min(coords_y[...,0]), np.min(coords_y[...,1])
        i_high_1, i_high_2 = np.max(coords_y[...,0]), np.max(coords_y[...,1])
        # build a meshgrid of coordinates based on coords_y
        y1i, y2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        # get the low and high indices of the x coordinates
        i_low_1, i_low_2 = np.min(coords_x[...,0]), np.min(coords_x[...,1])
        i_high_1, i_high_2 = np.max(coords_x[...,0]), np.max(coords_x[...,1])
        # build a meshgrid of coordinates based on coords_y
        x1i, x2i = np.meshgrid(np.linspace(i_low_1, i_high_1, n_grid), np.linspace(i_low_2, i_high_2, n_grid))

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=len(idx), figsize=(10 * len(idx), 6 * 4), sharex=True, squeeze=False)

        for col, idx_val in enumerate(idx):
            x_input_i = griddata(
                (coords_x[idx_val, :, 0], coords_x[idx_val, :, 1]), x[idx_val].detach().cpu().numpy(), (x1i, x2i), method='linear')
            y_true_i = griddata(
                (coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_true[idx_val], (y1i, y2i), method='linear')
            y_pred_i = griddata((coords_y[idx_val, :, 0], coords_y[idx_val, :, 1]), y_pred[idx_val], (y1i, y2i), method='linear')
            #y_true_i_norm = np.sqrt((1/(y_true_i.shape[0]*y_true_i.shape[1]))*np.sum(y_true_i**2))
            y_rel_diff_i = np.abs(y_pred_i - y_true_i) #/ np.abs(y_true_i + 1e-5)
            #y_rel_diff_i = np.abs(y_pred_i - y_true_i)

            #plot median and worst case relative error

            for i, ax in enumerate(axs[:, col]):
                if i == 0:
                    # plot input field x
                    im = ax.imshow(x_input_i, cmap='viridis')
                    ax.set_title(
                        f'Input Field (Index {idx_val})')

                if i == 1:
                    im = ax.imshow(y_true_i, cmap='viridis')
                    ax.set_title(
                        f'Ground Truth (Index {idx_val})')
                elif i == 2:
                    im = ax.imshow(y_pred_i, cmap='viridis')
                    ax.set_title(
                        f'Prediction (Index {idx_val})')
                elif i == 3:
                    # plot absolute relative error in log scale (difference divided by ground truth)
                    #im = ax.imshow(np.log10(y_rel_diff_i + 1e-10), cmap='viridis', vmin=-5, vmax=3)
                    im = ax.imshow(np.log10(y_rel_diff_i), cmap='inferno', vmin=-7,vmax=1)
                    ax.set_title(
                        f'Pointwise Error (Index {idx_val})')
                fig.colorbar(im, ax=ax)

        fig.suptitle(f'{tag} Predicted Fields: Prediction vs. Truth')
        plt.subplots_adjust(hspace=0.5)
        wandb.log(
            {f"plots/{tag}/Predicted Fields: Prediction vs. Truth": wandb.Image(fig)})
        plt.close()
