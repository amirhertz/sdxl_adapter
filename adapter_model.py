from __future__ import annotations

from custom_types import *
import options
from diffusers.models.attention import FeedForward
from diffusers.models import attention_processor
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline

scale_type = Union[float, int, Dict[str, float], Dict[str, T]]

Pipeline = Union[StableDiffusionPipeline, StableDiffusionXLPipeline]


class InvertDiffusion(nn.Module, ):
    def init_weights(self):
        return

    @torch.no_grad()
    def encode_text_sdxl(self, pipeline, prompt: str) -> tuple[dict[str, T], T]:

        def encode_(tokenizer, text_encoder):
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            text_embeddings_dict = text_encoder(tokens, output_hidden_states=True)
            pooled_prompt_embeds = text_embeddings_dict[0]
            text_embeddings = text_embeddings_dict.hidden_states[-2]
            if prompt == '':
                return torch.zeros_like(text_embeddings), torch.zeros_like(pooled_prompt_embeds)
            else:
                return text_embeddings, pooled_prompt_embeds

        if prompt not in self.text_to_embeddings:
            add_time_ids = pipeline._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float32,
                                                      text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim).to(
                self.device)
            prompt_embeds_1, _ = encode_(pipeline.tokenizer, pipeline.text_encoder)
            prompt_embeds_2, pooled_prompt_embeds_2 = encode_(pipeline.tokenizer_2, pipeline.text_encoder_2)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds_2, "time_ids": add_time_ids}
            prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)
            self.text_to_embeddings[prompt] = {key: added_cond_kwargs[key].detach().cpu() for key in
                                               added_cond_kwargs}, prompt_embeds.detach().cpu()
        else:
            added_cond_kwargs, prompt_embeds = self.text_to_embeddings[prompt]
            added_cond_kwargs = {key: added_cond_kwargs[key].to(pipeline.device) for key in added_cond_kwargs}
            prompt_embeds = prompt_embeds.to(pipeline.device)

        return added_cond_kwargs, prompt_embeds

    def encode_multi_text_sdxl(self, pipeline: Pipeline, text: Sequence[str]) -> tuple[dict[str, T], T]:
        if len(text) == 1:
            return self.encode_text_sdxl(pipeline, text[0])
        added_cond_kwargs, prompt_embeds = [], []
        for text_ in text:
            added_cond_kwargs_, prompt_embeds_ = self.encode_text_sdxl(pipeline, text_)
            added_cond_kwargs.append(added_cond_kwargs_)
            prompt_embeds.append(prompt_embeds_)
        return ({
                    key: torch.cat([item[key] for item in added_cond_kwargs])
                    for key in added_cond_kwargs[0]},
                torch.cat(prompt_embeds)
        )

    def set_lora_layers(self, pipeline: Pipeline):
        pipeline.unet.set_attn_processor(self.lora.as_dict())

    def clip_call(self, text_encoder, text_embeddings, input_ids) -> tuple[TN, T]:
        return None, text_embeddings

    @staticmethod
    def get_s_token_place(tokens: T):
        s_place = torch.where(tokens.eq(338))[1]
        assert tokens[0, s_place + 1] == 1844
        assert tokens[0, s_place + 2] == 49407
        return s_place

    @torch.no_grad()
    def encode_(self, tokens, text_encoder, trained_encoder=False, pretrained_embeddings=None) -> TS:
        text_embeddings_dict = text_encoder(tokens, output_hidden_states=True, return_dict=True)
        pooled_prompt_embeds = text_embeddings_dict[0]
        text_embeddings = text_embeddings_dict.hidden_states[-2]
        trained_embeddings = text_embeddings = text_embeddings.detach()
        return text_embeddings, trained_embeddings, pooled_prompt_embeds

    @torch.no_grad()
    def tokenize_single(self, prompt, tokenizer) -> TNS:
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return tokens, None

    @torch.no_grad()
    def tokenize(self, pipeline: StableDiffusionXLPipeline, prompt: str):
        tokens_1, _ = self.tokenize_single(prompt, pipeline.tokenizer)
        tokens_2, token_place = self.tokenize_single(prompt, pipeline.tokenizer_2)
        return tokens_1, tokens_2, token_place

    def init_text_embeddings(self, pipeline: StableDiffusionXLPipeline):
        if isinstance(pipeline, StableDiffusionXLPipeline):
            text_embeddings_1, trained_embeddings_1, _ = self.encode_(self.tokens_1, pipeline.text_encoder, True)
            text_embeddings_2, trained_embeddings_2, pooled_prompt_embeds_2 = self.encode_(self.tokens_2,
                                                                                           pipeline.text_encoder_2,
                                                                                           True)
            add_time_ids = pipeline._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), torch.float32,
                                                      text_encoder_projection_dim=pipeline.text_encoder_2.config.projection_dim).to(
                self.device)
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds_2, "time_ids": add_time_ids}
        else:
            raise NotImplementedError
        return text_embeddings_1, text_embeddings_2, trained_embeddings_1, trained_embeddings_2, added_cond_kwargs

    def inference_encode(self, pipeline, prompts: tuple[str, ...]) -> TS:
        tokens_1_all, text_embeddings_1_all = [], []
        tokens_2_all, text_embeddings_2_all, pooled_prompt_embeds_raw = [], [], []
        for prompt in prompts:
            tokens_1, tokens_2, token_place = self.tokenize(pipeline, prompt)
            text_embeddings_1, _, _ = self.encode_(tokens_1, pipeline.text_encoder, True,
                                                   pretrained_embeddings=self.trained_embeddings_1)
            text_embeddings_2, _, pooled_prompt_embeds = self.encode_(tokens_2, pipeline.text_encoder_2, True,
                                                                      pretrained_embeddings=self.trained_embeddings_2)
            tokens_1_all.append(tokens_1)
            tokens_2_all.append(tokens_2)
            text_embeddings_1_all.append(text_embeddings_1)
            text_embeddings_2_all.append(text_embeddings_2)
            pooled_prompt_embeds_raw.append(pooled_prompt_embeds)
        _, text_embeddings_1 = self.clip_call(pipeline.text_encoder, torch.cat(text_embeddings_1_all),
                                              torch.cat(tokens_1_all))

        pooled_prompt_embeds, text_embeddings_2 = self.clip_call(pipeline.text_encoder_2,
                                                                 torch.cat(text_embeddings_2_all),
                                                                 torch.cat(tokens_2_all))
        if pooled_prompt_embeds is None:
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_raw)
        return pooled_prompt_embeds, torch.cat((text_embeddings_1, text_embeddings_2), dim=-1)

    def prepare_text_embedding_inference(self,
                                         pipeline: Pipeline,
                                         prompts: tuple[str, ...]):
        batch_size = len(prompts)
        return self.encode_multi_text_sdxl(pipeline, [''] * batch_size + list(prompts))

    def inference_call(self, z, t, pipeline: Pipeline,
                       text_embeddings: T,
                       added_cond_kwargs: Dict[str, T],
                       scale: scale_type = 1., ) -> T:
        return pipeline.unet(z,
                             t,
                             encoder_hidden_states=text_embeddings,
                             cross_attention_kwargs={'scale': scale},
                             added_cond_kwargs=added_cond_kwargs,
                             return_dict=False)[0]

    def __call__(self, z, t, text, pipeline: Pipeline,
                 scale: scale_type = 1.,
                 ) -> T:
        return self.forward(z, t, text, pipeline, scale, )

    def forward(self, z, t, text: Sequence[str], pipeline: Pipeline, scale: scale_type = 1., ) -> T:

        added_cond_kwargs, text_embeddings = self.encode_multi_text_sdxl(pipeline, text)

        noise_pred = pipeline.unet(z,
                                   t,
                                   encoder_hidden_states=text_embeddings,
                                   added_cond_kwargs=added_cond_kwargs,
                                   cross_attention_kwargs={'scale': scale},
                                   return_dict=False)[0]
        return noise_pred

    def __init__(self, args: options.ArgsAdapter,
                 pipeline: Pipeline):
        super().__init__()
        self.args = args
        self.device = pipeline._execution_device
        self.text_to_embeddings: dict[str, tuple[dict[str, T], T]] = {}
        _ = self.encode_text_sdxl(pipeline, "")


def make_feed_forward_adapter(pipeline, rank):
    def get_feed_forward_layers(model, feed_forward_list):
        if isinstance(model, FeedForward):
            feed_forward_list.append(model)
        elif isinstance(model, nn.Module):
            for layer in model.children():
                get_feed_forward_layers(layer, feed_forward_list)

    def register_feed_forward_layer(layer):

        def forward_(hidden_states, scale=1.0):
            hidden_states = orig_forward(hidden_states)
            hidden_states = hidden_states + lora(hidden_states)
            return hidden_states

        out_features = layer.net[-1].out_features
        lora = attention_processor.LoRALinearLayer(out_features, out_features, rank)
        orig_forward = layer.forward
        layer.forward = forward_
        return lora

    feed_forward_layers = []
    lora_layers = []
    get_feed_forward_layers(pipeline.unet, feed_forward_layers)
    for layer in feed_forward_layers:
        lora_layers.append(register_feed_forward_layer(layer))
    return lora_layers


class Adapter(InvertDiffusion):

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, attention_processor.LoRALinearLayer):
                nn.init.normal_(layer.down.weight, std=1 / layer.rank)
                nn.init.zeros_(layer.up.weight)

    def __init__(self, args: options.ArgsAdapter, pipeline: Pipeline):
        super().__init__(args, pipeline)
        self.layers = nn.ModuleList(make_feed_forward_adapter(pipeline, args.rank))
