from __future__ import annotations
from custom_types import *
import options
from torch.optim import AdamW
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from utils import files_utils
from utils import image_loader
import adapter_model


@torch.no_grad()
def pipeline_call(
        pipeline: StableDiffusionXLPipeline,
        model: adapter_model.InvertDiffusion,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: TN = None,
        batch_size: int = 1,
        scale: float = 1.,
        prompts: Optional[List[str]] = None,
):
    # 0. Default height and width to unet
    height = pipeline.default_sample_size * pipeline.vae_scale_factor
    width = pipeline.default_sample_size * pipeline.vae_scale_factor
    device = pipeline._execution_device

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = pipeline.scheduler.timesteps
    added_cond_kwargs, prompt_embeds = model.prepare_text_embedding_inference(pipeline,  prompts=prompts)
    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = model.inference_call(latent_model_input, t, pipeline, prompt_embeds, added_cond_kwargs, scale=scale)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents.float()
    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type='np')
    return image


def init_diffusion():
    device = torch.device('cuda:0')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    scheduler.set_timesteps(50)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, variant="fp32", use_safetensors=True,
        scheduler=scheduler
    )
    pipeline = pipeline.to(device)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.vae.to(torch.float32)

    return pipeline


class FitAdapter:

    @torch.no_grad()
    def init_text_embedding(self, prompt: str) -> T:
        embeddings = self.pipeline._encode_prompt(prompt, self.device, 1, True).detach()
        null_embedding, text_embedding = embeddings.chunk(2, dim=0)
        text_embedding = text_embedding.repeat(self.args.batch_size, 1, 1)
        null_embedding = null_embedding.repeat(self.args.batch_size, 1, 1)
        return torch.cat((null_embedding, text_embedding), dim=0)

    @property
    def scheduler(self):
        return self.pipeline.scheduler

    def load_args(self, tag: str):
        args = files_utils.load_pickle(f'{self.get_save_path(tag)}/options.pkl')
        if args is None:
            raise ValueError
        return args

    def save_model(self, state, suffix=''):
        model, tag = state['model'], state['tag']
        if suffix:
            suffix = files_utils.add_prefix(suffix, '_')
        files_utils.save_model(model, f'{self.get_save_path(tag)}/model{suffix}.pt')
        files_utils.save_pickle(self.args, f'{self.get_save_path(tag)}/options.pkl')

    def load_model(self, tag, suffix='', strict: bool = True):
        if suffix:
            suffix = files_utils.add_prefix(suffix, '_')
            model_path = f'{self.get_save_path(tag)}/model{suffix}.pt'
        else:
            model_path = files_utils.collect(self.get_save_path(tag), '.pt')[-1]
            model_path = ''.join(model_path)
        files_utils.load_model(self.model, model_path, self.device, strict=strict)

    @torch.no_grad()
    def encode_image(self, image):
        return self.pipeline.vae.encode(image)['latent_dist'].mean * self.pipeline.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: T):
        latent = 1 / self.pipeline.vae.config.scaling_factor * latent
        image = self.pipeline.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return (image * 255).astype(np.uint8)

    def get_save_path(self, tag) -> str:
        return f'./checkpoints/{self.project_name}_{tag}'

    def get_image_save_path(self, tag) -> str:
        return f'{self.get_save_path(tag)}/images'

    def train_iter(self, state):
        state['log'] = {}
        model, optimizer = state['model'], state['optimizer']
        z, eps, t, y, text = self.prepare_data(state)

        noise_pred = model(
            z,
            t,
            text,
            self.pipeline,
        )
        loss = rec_loss = nnf.mse_loss(noise_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        state['lr_scheduler'].step()
        state['log']['loss'] = rec_loss.item()
        with torch.no_grad():
            k = len(self.args.image_path)
            state['z_0'] = self.pipeline.scheduler.step(noise_pred[:k], t[0].cpu().item(), z[:k, :4]).pred_original_sample
        return state

    def noise_input(self, z, eps, timestep: int):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t

    @torch.no_grad()
    def prepare_data(self, state) -> Tuple[T, ...]:
        image, text = state['data']
        b, c, h, w = image.shape
        encoded_image = self.encode_image(image.to(self.device))
        eps = torch.randn(*encoded_image.shape, device=self.device, generator=state['generator'])
        t = torch.randint(int(0), 1000, (b,), device=self.device, generator=state['generator']).long()
        encoded_image_noised = self.noise_input(encoded_image, eps, t)
        return encoded_image_noised, eps, t, encoded_image, text

    def get_trainable_model(self):
        return self.model

    def init_train_state(self, tag: str, ) -> Dict[str, Any]:
        trainable_model = self.get_trainable_model()
        param_groups = [{'params': trainable_model.parameters(), 'betas': (.9, .999),
                         'lr': self.args.lr, 'eps': 1e-8,
                         'weight_decay': 1e-2}]
        optimizer = AdamW(param_groups)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.num_training_steps,
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(1234)
        state = {
            'lr_scheduler': lr_scheduler,
            'optimizer': optimizer,
            'model': trainable_model,
            'generator': generator,
            'tag': tag,
        }
        return state

    def between_train_iters(self, state, i):
        if (i + 1) % self.args.save_checkpoint_every == 0:
            self.save_model(state)
        return state

    def train(self, tag: str, args=None):

        if args:
            self.args = args
        if self.model is None:
            self.model = self.init_generator()
        self.model.init_weights()
        state = self.init_train_state(tag)
        bar = tqdm(total=self.args.num_training_steps, desc=f"{state['tag']}")
        dataset = self.get_ds(True)
        dataloader = iter(DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True))
        for i in range(self.args.num_training_steps):
            state['iter'] = i
            state['data'] = next(dataloader)
            state = self.train_iter(state)
            bar.set_postfix(state['log'])
            bar.update()
            state = self.between_train_iters(state, i)

    def init_generator(self) -> adapter_model.InvertDiffusion:
        return adapter_model.Adapter(self.args, self.pipeline).to(self.device)

    def get_ds(self, is_train: bool) -> image_loader.ImageDS:
        return image_loader.ImageDS(self.args, is_train)

    @torch.no_grad()
    def generate(self, content: tuple[str, ...], cfg=7.5, seed=-1,
                 latents=None) -> ARRAY:
        generator = torch.Generator()
        if seed < 0:
            seed = int(np.random.randint(0, 100000, (1,)))
        generator.manual_seed(seed)
        outputs = []
        prompts = [f'{item} {self.args.style_desc}' for item in content]
        if latents is not None:
            latents = latents.to(self.device)
        for i in range(0, len(prompts), 1):
            # images = next(dl)
            if latents is not None:
                latent = latents[i: i + 1]
            else:
                latent = None
            output = pipeline_call(self.pipeline, self.model, guidance_scale=cfg,
                                   generator=generator, scale=1., batch_size=1,
                                   prompts=[prompts[i]], latents=latent)
            outputs.append(output)
        outputs = np.concatenate(outputs)
        outputs = (outputs.clip(0, 1) * 255).astype(np.uint8)
        # if DEBUG:
        output = files_utils.images_to_grid(outputs, num_rows=max(1, min(4, outputs.shape[0] // 4)))
        files_utils.show_images(output)
        return outputs

    def __init__(self, args: Optional[options.ArgsAdapter] = None, tag: Optional[str] = None,
                 project_name: str = "feature_fusion",
                 ):
        self.project_name = project_name
        assert not (args is None and tag is None)
        if tag:
            self.args = self.load_args(tag)
        else:
            self.args = args
        self.device = torch.device('cuda:0')
        self.pipeline = init_diffusion()
        self.model = self.init_generator()
        if tag:
            self.load_model(tag)
        with torch.inference_mode():
            self.alphas = torch.sqrt(self.pipeline.scheduler.alphas_cumprod).to(self.device)
            self.sigmas = torch.sqrt(1 - self.pipeline.scheduler.alphas_cumprod).to(self.device)


def train():
    image_path = './images/house_3d.png'
    content_desc = 'a house'
    style_desc = 'in simple 3d render style'

    args = options.ArgsAdapter(lr=2e-4,
                               rank=16,
                               model_type=options.ModelType.Adapter,
                               num_training_steps=1000,
                               image_path=image_path,
                               style_desc=style_desc,
                               content_desc=content_desc,
                               save_checkpoint_every=1000,
                               warmup_steps=100,
                               )
    trainer = FitAdapter(args)
    exp_name = 'house'
    trainer.train(exp_name)


def generate(exp_name):
    trainer = FitAdapter(None, exp_name)
    trainer.generate(('a camel', 'a guitar', 'a balloon', 'a temple'))



def main():
    train()
    generate('house')


if __name__ == '__main__':
    main()
