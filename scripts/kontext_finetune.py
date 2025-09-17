from diffusers.models import AutoencoderKL
from src.utils.train_utils import instantiate_from_config
from accelerate import Accelerator
import torch
from tqdm import tqdm
import os
from src.utils.image_utils import save_image_tensor
from src.utils.general_utls import log_captions
import numpy as np
import inspect
from transformers import CLIPTokenizer, T5Tokenizer, CLIPTextModel, T5EncoderModel
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from diffusers.utils import convert_unet_state_dict_to_peft
from src.models.coto.coto_helper import CoToHelper
from peft.tuners.lora import LoraLayer
from src.models.adaptive_sampler import AdaptiveSampler


"""
Fine-tuning script for the Flux Kontext model.
Supports three image conditions and one text condition.
Can also function as a Text-to-Image (T2I) model when all three image conditions are set to None.
"""

class ContextTrainer:
    def __init__(
            self,
            model_directory,
            scheduler_config
    ):
        transformer_model = FluxTransformer2DModel.from_pretrained(
            model_directory, subfolder='transformer'
        )
        primary_tokenizer = CLIPTokenizer.from_pretrained(
            model_directory,
            subfolder="tokenizer"
        )
        secondary_tokenizer = T5Tokenizer.from_pretrained(
            model_directory,
            subfolder="tokenizer_2"
        )
        primary_text_encoder = CLIPTextModel.from_pretrained(
            model_directory, subfolder="text_encoder"
        )
        secondary_text_encoder = T5EncoderModel.from_pretrained(
            model_directory, subfolder="text_encoder_2"
        )

        self.scheduler = instantiate_from_config(scheduler_config)
        self.scheduler_config = scheduler_config
        self.latent_encoder = AutoencoderKL.from_pretrained(model_directory, subfolder='vae')
        self.precision_dtype = torch.float32
        self.primary_tokenizer = primary_tokenizer
        self.secondary_tokenizer = secondary_tokenizer
        self.primary_text_encoder = primary_text_encoder
        self.secondary_text_encoder = secondary_text_encoder
        self.transformer_model = transformer_model

    def execute_training(self, training_loader, validation_loader, training_config, output_dir):
        self.train_guidance = training_config.guidance.training_scale
        self.inference_guidance = training_config.guidance.inference_scale
        self.max_seq_length = training_config.max_sequence_length
        self.validate_on_checkpoint = training_config.logging.validate_on_checkpoint
        self.coto_total_steps = training_config.coto.coto_total_steps
        use_lora_adapter = training_config.adapter.use_lora

        # Here, the batch size is set to 1 because we train with different aspect ratios.
        # To mitigate the impact of a small batch size, we set a large accumulation step
        assert training_loader.batch_size == 1

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        accelerator = Accelerator(
            mixed_precision=training_config.precision_mode,
            gradient_accumulation_steps=training_config.gradient_accumulation
        )

        if training_config.get('pretrained_model', None):
            self.transformer_model.load_state_dict(
                torch.load(training_config.pretrained_model, map_location='cpu'),
                strict=False
            )
            accelerator.print(f'Loaded pretrained model from {training_config.pretrained_model}')

        device = accelerator.device

        self.latent_encoder.enable_slicing()
        self.latent_encoder.eval()
        self.latent_encoder.requires_grad_(False)
        self.primary_text_encoder.eval()
        self.secondary_text_encoder.eval()
        self.primary_text_encoder.requires_grad_(False)
        self.secondary_text_encoder.requires_grad_(False)

        if use_lora_adapter:
            self.transformer_model.requires_grad_(False)
            lora_configuration = LoraConfig(
                r=training_config.adapter.lora_rank,
                lora_alpha=training_config.adapter.lora_rank,
                init_lora_weights="gaussian",
                target_modules=[
                    "to_k", "to_q", "attn.to_v", 'ff.net.2',
                    'to_out.0', 'norm1.linear', 'norm.linear',
                    'proj_mlp', 'ff.net.0.proj'
                ],
            )
            accelerator.print(f'LoRA configuration: {lora_configuration}')
            self.transformer_model.add_adapter(lora_configuration)

            if training_config.get("pretrained_lora", None):
                lora_state = FluxPipeline.lora_state_dict(training_config.pretrained_lora)
                transformer_state = {
                    k.replace("transformer.", ""): v
                    for k, v in lora_state.items()
                    if k.startswith("transformer.")
                }
                transformer_state = convert_unet_state_dict_to_peft(transformer_state)
                incompatible = set_peft_model_state_dict(
                    self.transformer_model, transformer_state, adapter_name="default"
                )

                if incompatible is not None:
                    unexpected_keys = getattr(incompatible, "unexpected_keys", None)
                    if unexpected_keys:
                        accelerator.print(
                            f"Found unexpected keys when loading adapter weights: {unexpected_keys}"
                        )
                accelerator.print(f"Loaded pretrained LoRA from {training_config.pretrained_lora}")
        else:
            self.transformer_model.requires_grad_(True)

        if training_config.gradient_checkpointing:
            self.transformer_model.enable_gradient_checkpointing()

        if use_lora_adapter:
            optimizable_params = []
            optimizable_param_names = []
            for name, param in self.transformer_model.named_parameters():
                if param.requires_grad:
                    optimizable_params.append(param)
                    optimizable_param_names.append(name)
            accelerator.print("\nTrainable LoRA Parameters:")
            accelerator.print("=" * 60)
            for i, param_name in enumerate(optimizable_param_names):
                accelerator.print(f"  {i+1:3d}. {param_name}")
            accelerator.print("-" * 60)
            accelerator.print(f"  Total: {len(optimizable_param_names)} LoRA weights")
            accelerator.print("=" * 60)
        else:
            optimizable_params = self.transformer_model.parameters()

        optimizer = torch.optim.AdamW(
            optimizable_params,
            lr=training_config.optimizer.learning_rate,
            weight_decay=training_config.optimizer.weight_decay,
            betas=(training_config.optimizer.beta1, training_config.optimizer.beta2)
        )

        prepared_components = accelerator.prepare(
            self.transformer_model, optimizer, training_loader, validation_loader
        )
        self.transformer_model, optimizer, training_loader, validation_loader = prepared_components

        if accelerator.mixed_precision == "fp16":
            precision_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            precision_dtype = torch.bfloat16
        else:
            precision_dtype = torch.float32

        self.precision_dtype = precision_dtype
        self.latent_encoder.to(device, dtype=self.precision_dtype)
        self.primary_text_encoder.to(device, dtype=self.precision_dtype)
        self.secondary_text_encoder.to(device, dtype=self.precision_dtype)

        if use_lora_adapter and training_config.coto.coto_total_steps > 0:
            lora_modules = [
                module for name, module in self.transformer_model.named_modules()
                if isinstance(module, LoraLayer)
            ]
            accelerator.print(f'Coto configured {len(lora_modules)} LoRA modules')
            context_helper = CoToHelper(lora_modules)
        else:
            context_helper = None
            accelerator.print('Using full parameter training')

        if accelerator.is_local_main_process:
            print("\n" + "Training Configuration")
            print("=" * 40)
            print(f"  Training examples: {len(training_loader.dataset):>12,}")
            print(f"  Batches per epoch: {len(training_loader):>13,}")
            print(f"  Total epochs:      {training_config.num_train_epochs:>13,}")
            print("=" * 40 + "\n")
        global_step = 0
        if context_helper is not None:
            context_helper.on_train_begin(training_config.coto.coto_total_steps)

        noise_scheduler = instantiate_from_config(self.scheduler_config)
        self.timestep_losses = torch.zeros_like(noise_scheduler.timesteps).to(torch.float32).to(device)
        self.sampler = AdaptiveSampler(np.load(training_config.adaptive_sampling.initial_sampling_weights), smoothing_factor=0.5)

        for epoch_idx in range(training_config.num_train_epochs):
            cumulative_loss = 0
            progress_bar = tqdm(training_loader, disable=not accelerator.is_local_main_process)

            for batch_idx, batch_data in enumerate(progress_bar):
                self.transformer_model.train()
                with accelerator.accumulate([self.transformer_model]):
                    processed_input = self.prepare_batch_input(batch_data, accelerator, return_reconstruction=False)
                    target_latents = processed_input['target']
                    pooled_states = processed_input['state_pool']
                    hidden_states = processed_input['state']
                    img_ids = processed_input['img_ids']
                    guidance = processed_input['guidance']
                    text_ids = processed_input['text_ids']
                    img1 = processed_input['img1']
                    img2 = processed_input['img2']
                    img3 = processed_input['img3']

                    noise = torch.randn_like(target_latents)
                    batch_size = noise.shape[0]

                    sampling_indices = self.sampler.sample_with_power_weighting(batch_size, exponent=3.0)
                    indices_tensor = torch.tensor(sampling_indices, dtype=torch.long, device='cpu')

                    selected_timesteps = noise_scheduler.timesteps[indices_tensor].to(device=accelerator.device)
                    sigma_values = compute_sigma(
                        selected_timesteps, accelerator.device, noise_scheduler,
                        n_dim=noise.ndim, dtype=noise.dtype
                    )
                    noisy_latents = sigma_values * noise + (1.0 - sigma_values) * target_latents

                    batch, channels, height, width = noisy_latents.shape

                    packed_noisy_latents = pack_latents_representation(
                        noisy_latents,
                    ).to(self.precision_dtype)

                    packed_latents = self._prepare_condition_latents(img1, img2, img3)
                    packed_latents = torch.cat([packed_noisy_latents] + packed_latents, dim=1)
                    model_prediction = self.transformer_model(
                        hidden_states=packed_latents,
                        timestep=selected_timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_states,
                        encoder_hidden_states=hidden_states,
                        txt_ids=text_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                    model_prediction = model_prediction[:, : packed_noisy_latents.size(1)]
                    model_prediction = unpack_latent_representation(
                        model_prediction,
                        height=height,
                        width=width
                    )

                    loss_per_sample = torch.mean(
                        ((model_prediction.float() - (noise - target_latents).float()) ** 2).reshape(target_latents.shape[0], -1),
                        1,
                    )

                    with torch.no_grad():
                        self.timestep_losses[indices_tensor.to(device)] = loss_per_sample

                    total_loss = loss_per_sample.mean()
                    cumulative_loss += total_loss.detach().item()
                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(optimizable_params, training_config.max_gradient_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    gathered_losses = accelerator.gather(self.timestep_losses[None])
                    mean_losses = gathered_losses.mean(dim=0, keepdim=False)
                    self.sampler.update_distribution(mean_losses)
                    self.timestep_losses.zero_()

                    if global_step % training_config.logging.log_frequency == 0:
                        if accelerator.is_local_main_process:
                            self.sampler.export_weights(os.path.join(output_dir, f'{global_step}_sampling_weights'))
                            self.sampler.export_statistics(os.path.join(output_dir, f'{global_step}_sampling_updates'))

                        self.transformer_model.eval()
                        sample_output = self.generate_sample(batch_data, training_config.log_image_cfg, accelerator)
                        self._save_result(sample_output, output_dir, 'train', global_step, local_rank)
                    if accelerator.is_local_main_process:
                        progress_bar.set_description(
                            f"Epoch={epoch_idx}, Step={global_step}, Loss={cumulative_loss/(batch_idx+1):.5f}"
                        )

                        if global_step % training_config.logging.checkpoint_frequency == 0 and global_step != 0:
                            if use_lora_adapter:
                                save_path = os.path.join(output_dir, f"step_{global_step}")
                                unwrapped_model = accelerator.unwrap_model(self.transformer_model)
                                lora_layers = get_peft_model_state_dict(unwrapped_model)
                                FluxPipeline.save_lora_weights(
                                    save_directory=save_path,
                                    transformer_lora_layers=lora_layers,
                                )
                            else:
                                model_path = os.path.join(output_dir, f"{global_step}_transformer.pt")
                                self.save_model_checkpoint(model_path, accelerator.unwrap_model(self.transformer_model), accelerator)

                    if self.validate_on_checkpoint and global_step % training_config.logging.checkpoint_frequency == 0 and global_step != 0:
                        self.transformer_model.eval()
                        for val_idx, val_batch in enumerate(validation_loader):
                            sample_output = self.generate_sample(val_batch, training_config.log_image_cfg, accelerator)
                            self._save_result(sample_output, output_dir, 'val', global_step, local_rank, current_step=val_idx)

                    global_step += 1
                    if context_helper is not None:
                        context_helper.on_step_end(global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            if use_lora_adapter:
                final_save_path = os.path.join(output_dir, "final")
                unwrapped_model = accelerator.unwrap_model(self.transformer_model)
                final_lora_layers = get_peft_model_state_dict(unwrapped_model)
                FluxPipeline.save_lora_weights(
                    save_directory=final_save_path,
                    transformer_lora_layers=final_lora_layers,
                )
            else:
                final_model_path = os.path.join(output_dir, "final_transformer.pt")
                self.save_model_checkpoint(final_model_path, accelerator.unwrap_model(self.transformer_model), accelerator)

    def prepare_batch_input(self, batch, accelerator, return_reconstruction):
        target = batch['target']
        caption = batch['prompt']
        img1 = batch.get('img1', None)
        img2 = batch.get('img2', None)
        img3 = batch.get('img3', None)


        bsz = target.shape[0]
        train_guidance = torch.tensor([self.train_guidance], device=accelerator.device)
        train_guidance = train_guidance.expand(bsz)
        inference_guidance = torch.tensor([self.inference_guidance], device=accelerator.device)
        inference_guidance = inference_guidance.expand(bsz)
        prepared = {}
        with torch.no_grad():
            state, state_pool, text_ids = compute_text_embeddings(caption, [self.primary_text_encoder, self.secondary_text_encoder],
                                                                  [self.primary_tokenizer, self.secondary_tokenizer],
                                                                  self.max_seq_length, accelerator.device)
            latent_encoder = accelerator.unwrap_model(self.latent_encoder)

            z_target = self._encode_img(target, latent_encoder)
            z_img1 = self._encode_img(img1, latent_encoder)
            z_img2 = self._encode_img(img2, latent_encoder)
            z_img3 = self._encode_img(img3, latent_encoder)

            if return_reconstruction:
                target_rec = self._decode_img(z_target, latent_encoder)
                img1_rec = self._decode_img(z_img1, latent_encoder)
                img2_rec = self._decode_img(z_img2, latent_encoder)
                img3_rec = self._decode_img(z_img3, latent_encoder)
                prepared['target_rec'] = target_rec
                prepared['img1_rec'] = img1_rec
                prepared['img2_rec'] = img2_rec
                prepared['img3_rec'] = img3_rec

        target_ids = self._prepare_image_ids(z_target, 0)
        img1_ids = self._prepare_image_ids(z_img1, 1)
        img2_ids = self._prepare_image_ids(z_img2, 1)
        img3_ids = self._prepare_image_ids(z_img3, 1)
        if img1_ids is not None and img2_ids is not None:
            img2_ids[:, 2] += z_img1.shape[-1] // 2

        if img2_ids is not None and img3_ids is not None:
            img3_ids[:, 2] = img3_ids[:, 2] + z_img1.shape[-1] // 2 + z_img2.shape[-1] // 2

        ids_all = target_ids
        if img1_ids is not None:
            ids_all = torch.cat([ids_all, img1_ids], dim=0)
        if img2_ids is not None:
            ids_all = torch.cat([ids_all, img2_ids], dim=0)
        if img3_ids is not None:
            ids_all = torch.cat([ids_all, img3_ids], dim=0)


        prepared['img1'] = z_img1
        prepared['img2'] = z_img2
        prepared['img3'] = z_img3
        prepared['target'] = z_target
        prepared['state_pool'] = state_pool
        prepared['state'] = state
        prepared['img_ids'] = ids_all
        prepared['guidance'] = train_guidance
        prepared['guidance_infer'] = inference_guidance
        prepared['text_ids'] = text_ids
        prepared['caption'] = caption
        return prepared

    def save_model_checkpoint(self, path, model, accelerator):
        model_state = accelerator.get_state_dict(model)
        accelerator.save(model_state, path)

    def generate_sample(self, batch, sampling_config, accelerator):
        scheduler = instantiate_from_config(self.scheduler_config)
        inference_steps = sampling_config.steps
        model = accelerator.unwrap_model(self.transformer_model)

        with torch.no_grad():
            prepared = self.prepare_batch_input(batch, accelerator, return_reconstruction=True)
            target_latent = prepared['target']
            state_pool = prepared['state_pool']
            state = prepared['state']
            text_ids = prepared['text_ids']
            img_ids = prepared['img_ids']
            guidance = prepared['guidance_infer']
            img1 = prepared['img1']
            img2 = prepared['img2']
            img3 = prepared['img3']

            noise_latent = torch.randn_like(target_latent)
            bsz, c, h, w = noise_latent.shape
            sigmas = np.linspace(1.0, 1 / inference_steps, inference_steps)
            shift_value = compute_shift_amount(
                h * w // 4,
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )

            timesteps, num_steps = get_inference_timesteps(
                scheduler,
                inference_steps,
                accelerator.device,
                None,
                sigmas,
                mu=shift_value,
            )
            img = pack_latents_representation(noise_latent)
            packed_latents = self._prepare_condition_latents(img1, img2, img3)
            for t in timesteps:
                timestep = t.expand(bsz).to(accelerator.device)
                noise_pred = model(
                    hidden_states=torch.cat([img] + packed_latents, dim=1),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=state_pool,
                    encoder_hidden_states=state,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : img.size(1)]
                img = scheduler.step(noise_pred, t, img).prev_sample
            latent_encoder = accelerator.unwrap_model(self.latent_encoder)
            img = unpack_latent_representation(img, h, w)
            img = img / latent_encoder.config.scaling_factor + latent_encoder.config.shift_factor
            img = latent_encoder.decode(img).sample.to(torch.float32)
            sample_result = {}
            sample_result['result'] = img
            sample_result['target_rec'] = prepared['target_rec']
            sample_result['img1_rec'] = prepared['img1_rec']
            sample_result['img2_rec'] = prepared['img2_rec']
            sample_result['img3_rec'] = prepared['img3_rec']
            sample_result['caption'] = prepared['caption']
        return sample_result

    def _encode_img(self, img, latent_encoder):
        if img is not None:
            z_img = (latent_encoder.encode(img.to(self.precision_dtype)).latent_dist.mode() - latent_encoder.config.shift_factor) * latent_encoder.config.scaling_factor
        else:
            z_img = None
        return z_img


    def _decode_img(self, img_latent, latent_encoder):
        if img_latent is not None:
            img_rec = latent_encoder.decode(img_latent / latent_encoder.config.scaling_factor + latent_encoder.config.shift_factor).sample
        else:
            img_rec = None
        return img_rec

    def _prepare_image_ids(self, img_latent, img_index):
        if img_latent is None:
            return None
        b, c, height, width = img_latent.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        # latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids[..., 0] = img_index
        return latent_image_ids.to(device=img_latent.device, dtype=img_latent.dtype)

    def _prepare_condition_latents(self, img1, img2, img3):

        latents = [img1, img2, img3]
        latents = [pack_latents_representation(i) for i in latents if i is not None]
        return latents

    def _save_result(self, sample_output, output_dir, tag, global_step, local_rank, current_step=None):
        if current_step is None:
            current_step = ''
        else:
            current_step = '{}_'.format(current_step)
        save_image_tensor(
            sample_output['result'],
            os.path.join(output_dir, tag, f'{global_step}_{current_step}{local_rank}_result.jpg')
        )
        save_image_tensor(
            sample_output['target_rec'],
            os.path.join(output_dir, tag, f'{global_step}_{current_step}{local_rank}_target.jpg')
        )
        save_image_tensor(
            sample_output['img1_rec'],
            os.path.join(output_dir, tag, f'{global_step}_{current_step}{local_rank}_img1.jpg')
        )
        save_image_tensor(
            sample_output['img2_rec'],
            os.path.join(output_dir, tag, f'{global_step}_{current_step}{local_rank}_img2.jpg')
        )
        save_image_tensor(
            sample_output['img3_rec'],
            os.path.join(output_dir, tag, f'{global_step}_{current_step}{local_rank}_img3.jpg')
        )
        log_captions(sample_output['caption'], os.path.join(output_dir, tag, '{}_{}{}_cap.txt'.format(global_step, current_step, local_rank)))

def get_inference_timesteps(
        scheduler,
        num_inference_steps = None,
        device = None,
        timesteps = None,
        sigmas = None,
        **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def compute_sigma(timesteps, device, scheduler, n_dim=4, dtype=torch.float32):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def compute_shift_amount(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def pack_latents_representation(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents

def unpack_latent_representation(latents, height, width):
    batch_size, num_patches, channels = latents.shape

    latents = latents.view(batch_size, height//2, width//2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents







def compute_text_embeddings(prompt, text_encoders, tokenizers, max_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids

def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds
