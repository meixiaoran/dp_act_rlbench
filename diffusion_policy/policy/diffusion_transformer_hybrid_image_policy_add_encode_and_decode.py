from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.diffusion.udit_models import U_DiT_DP
from diffusion_policy.model.diffusion.dic_models import DiC_S
from diffusion_policy.model.diffusion.dic_model_B import DiC_B
from diffusion_policy.model.diffusion.dit_model_end import DiT_B_4, DiT_S_4, LatentActionEncoder, LatentActionDecoder
from diffusion_policy.model.diffusion.j_dit import JiT_B_16

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 # task params
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 # image
                 crop_shape=(100, 100),
                 obs_encoder_group_norm=False,
                 eval_fixed_crop=False,
                 # arch
                 n_layer=8,
                 n_cond_layers=0,
                 n_head=4,
                 n_emb=256,
                 p_drop_emb=0.0,
                 p_drop_attn=0.3,
                 causal_attn=True,
                 time_as_cond=True,
                 obs_as_cond=True,
                 pred_action_steps_only=False,
                 # parameters passed to step
                 **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']  # 8
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_shape = next(iter(obs_key_shapes.items()))[1]
        print(rgb_shape)
        # resnet = get_resnet("resnet18", input_shape = rgb_shape)
        resnet = get_resnet("resnet18")
        obs_encoder = MultiImageObsEncoder(shape_meta, resnet, resize_shape = None, crop_shape = (100, 100),
                                           random_crop= True,use_group_norm= True,
                                           share_rgb_model = False, imagenet_norm = False)
        # use_group_norm = True (必选)

        obs_feature_dim = obs_encoder.output_shape()[0]
        print("obs_feature_dim" + str(obs_feature_dim))

        # create diffusion model
        obs_feature_dim = obs_feature_dim
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        # model = TransformerForDiffusion(
        #     input_dim=input_dim,  # 8
        #     output_dim=output_dim,
        #     horizon=horizon,
        #     n_obs_steps=n_obs_steps,
        #     cond_dim=cond_dim,
        #     n_layer=n_layer,
        #     n_head=n_head,
        #     n_emb=n_emb,
        #     p_drop_emb=p_drop_emb,
        #     p_drop_attn=p_drop_attn,
        #     causal_attn=causal_attn,
        #     time_as_cond=time_as_cond,
        #     obs_as_cond=obs_as_cond,
        #     n_cond_layers=n_cond_layers
        # )
        # model = DiC_S()

        latent_action_encode = LatentActionEncoder(8, 4 * 8)
        latent_action_decode = LatentActionDecoder(4 * 8)

        model = DiT_S_4()
        self.model = nn.ModuleDict({
            'latent_action_encode': latent_action_encode,
            'latent_action_decode': latent_action_decode,
            'obs_encoder': obs_encoder,
            'model': model
        })

        # self.obs_encoder = obs_encoder
        # self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model['model'].parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.model['obs_encoder'].parameters()))

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           cond=None, generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. predict model output
            # t = t.view(-1)  # 变成 shape (1,)
            # t = t.to('cuda:0')
            
            model_output = model['model'](trajectory, t, cond)
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # def get_optimizer(
    #         self,
    #         transformer_weight_decay: float,
    #         obs_encoder_weight_decay: float,
    #         learning_rate: float,
    #         betas: Tuple[float, float]
    # ) -> torch.optim.Optimizer:
    #     optim_groups = self.model['model'].get_optim_groups(
    #         weight_decay=transformer_weight_decay)
    #     optim_groups.append({
    #         "params": self.model['obs_encoder'].parameters(),
    #         "weight_decay": obs_encoder_weight_decay
    #     })
    #     optimizer = torch.optim.AdamW(
    #         optim_groups, lr=learning_rate, betas=betas
    #     )
    #     return optimizer
    def compute_latent_diffusion_loss(
            self,
            action_raw,                # (B, T, A)
            latent_action_raw,         # (B, T, D)
            noised_latent_action,      # (B, T, D)
            latent_action_pred,        # (B, T, D)
            pred_action,               # (B, T, A)
            loss_mask=None,            # (B, T) optional
            lambda_smooth=0.1,
            lambda_recon=0.1,
            lambda_var=0.1,
            sigma_min=0.05
        ):
        """
        Compute the combined loss for latent diffusion policy.
        """

        # 1️⃣ Diffusion loss in latent space
        loss_diff = F.mse_loss(latent_action_pred, latent_action_raw, reduction='none')  # (B, T, D)
        
        loss_diff = loss_diff.mean()

        # 2️⃣ Temporal smoothness on latent
        loss_smooth = ((latent_action_pred[:, 1:] - latent_action_pred[:, :-1]) ** 2).mean()

        # 3️⃣ Reconstruction loss in action space
        loss_recon = F.mse_loss(pred_action, action_raw, reduction='none')
        if loss_mask is not None:
            loss_mask = loss_mask.type(loss_recon.dtype)
            loss_recon = loss_recon * loss_mask
        loss_recon = loss_recon.mean()

        # 4️⃣ Latent variance / isotropy loss (prevent collapse)
        latent_std = latent_action_pred.std(dim=(0,1))  # over batch and time
        loss_var = torch.sum(F.relu(sigma_min - latent_std))
        
        # 5️⃣ Combine losses
        loss = loss_diff \
            + lambda_smooth * loss_smooth \
            + lambda_recon * loss_recon \
            + lambda_var * loss_var

        return loss, {
            "loss_diff": loss_diff.item(),
            "loss_smooth": loss_smooth.item(),
            "loss_recon": loss_recon.item(),
            "loss_var": loss_var.item()
        }


    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)
        
        action_raw = trajectory

        trajectory, k1 = self.model['latent_action_encode'](trajectory)

        latent_action_raw = trajectory
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        noised_latent_action = noisy_trajectory

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        # noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        latent_action_pred = self.model['model'](noisy_trajectory, timesteps, cond)
        
        pred_action = self.model['latent_action_decode'](latent_action_pred)



        loss, loss_list = self.compute_latent_diffusion_loss(
        action_raw,
        latent_action_raw,
        noised_latent_action,
        latent_action_pred,
        pred_action,
        loss_mask
        )

        
        return loss


        # pred, latent_action_raw, latent_action_pred = self.model['model'](noisy_trajectory, timesteps, cond)

        # # 1️⃣ Diffusion MSE loss
        # loss_diff = F.mse_loss(pred, noise, reduction='none')       # (B, T, A)
        # loss_diff = loss_diff * loss_mask.type(loss_diff.dtype)
        # loss_diff = reduce(loss_diff, 'b ... -> b (...)', 'mean')
        # loss_diff = loss_diff.mean()

        # # 2️⃣ Temporal smoothness on latent
        # # latent_action_pred shape: (B, T, D)
        # loss_smooth = ((latent_action_pred[:, 1:] - latent_action_pred[:, :-1]) ** 2).mean()

        # # 3️⃣ Optional: latent reconstruction loss
        # # latent_action_raw: encoder output, latent_action_pred: DiT output (denoised)
        # loss_recon = ((latent_action_pred - latent_action_raw) ** 2).mean()

        # # 4️⃣ Optional: latent variance / isotropy loss
        # # Encourage each latent dimension to have sufficient std
        # latent_std = latent_action_pred.std(dim=(0,1))   # over batch and time
        # sigma_min = 0.05
        # loss_var = torch.sum(F.relu(sigma_min - latent_std))

        # # 5️⃣ Combine
        # lambda_smooth = 0.1
        # lambda_recon = 0.1
        # lambda_var = 0.1

        # loss = loss_diff \
        #     + lambda_smooth * loss_smooth \
        #     + lambda_recon * loss_recon \
        #     + lambda_var * loss_var

        # return loss

