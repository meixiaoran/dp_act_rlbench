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
from diffusion_policy.model.diffusion.dit_model_raw import DiT_B_4, DiT_S_4
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
        model = DiT_S_4()
        self.model = nn.ModuleDict({
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
    
    def _t_cont_to_index(self, t_cont: torch.Tensor) -> torch.Tensor:
        # t_cont: (B,) in [0, 1]
        # 复用原 scheduler 的 num_train_timesteps 作为 embedding 分辨率
        N = self.noise_scheduler.config.num_train_timesteps
        t_idx = torch.clamp((t_cont * (N - 1)).round().long(), 0, N - 1)
        return t_idx


    @torch.no_grad()
    def conditional_sample_flow(self,
                                condition_data, condition_mask,
                                cond=None, generator=None,
                                method="euler",
                                **kwargs):
        model = self.model['model']

        x = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        B = x.shape[0]
        steps = 10
        dt = 1.0 / steps

        for i in range(steps):
            # 连续时间 t in [0,1)
            t_cont = torch.full((B,), i / steps, device=x.device, dtype=torch.float32)
            t_idx = self._t_cont_to_index(t_cont)

            # 1) enforce conditioning
            x[condition_mask] = condition_data[condition_mask]

            # 2) predict velocity
            v = model(x, t_idx, cond)  # v_theta

            # 3) ODE step
            x = x + dt * v

        # finally enforce conditioning
        x[condition_mask] = condition_data[condition_mask]
        return x


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
        nsample = self.conditional_sample_flow(
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

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        cond = None
        trajectory = nactions  # x1

        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            cond = nobs_features.reshape(batch_size, To, -1)

            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]  # x1
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.model['obs_encoder'](this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()  # x1

        # condition mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        loss_mask = ~condition_mask  # only train on unknown parts

        # Flow Matching endpoints
        x1 = trajectory
        x0 = torch.randn_like(x1)  # noise endpoint

        # sample continuous t ~ U(0,1)
        t_cont = torch.rand((batch_size,), device=x1.device, dtype=torch.float32)
        t_view = t_cont.view(batch_size, *([1] * (x1.ndim - 1)))  # broadcast to (B,1,1,...)

        # interpolate
        xt = (1.0 - t_view) * x0 + t_view * x1

        # apply conditioning (inpainting)
        xt[condition_mask] = x1[condition_mask]

        # target velocity
        target_v = (x1 - x0)

        # model predicts velocity
        t_idx = self._t_cont_to_index(t_cont)  # integer timesteps for embedding reuse
        pred_v = self.model['model'](xt, t_idx, cond)

        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
        return loss
