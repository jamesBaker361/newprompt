from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch
from text_embedding_helpers import prepare_textual_inversion

class BetterDefaultDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self,train_text_encoder:bool,
                 train_text_encoder_embeddings:bool,
                 train_unet:bool,
                  use_lora_text_encoder:bool, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sd_pipeline.vae.requires_grad_(False)
        if train_text_encoder and train_text_encoder_embeddings:
            raise Exception("train text encoder OR embedding!!!")
        elif train_text_encoder:
            if use_lora_text_encoder:
                self.sd_pipeline.text_encoder.requires_grad_(False)
                text_encoder_target_modules=["q_proj", "v_proj"]
                text_encoder_config=LoraConfig(
                    r=4,
                    lora_alpha=16,
                    target_modules=text_encoder_target_modules,
                    lora_dropout=0.0
                )
                self.sd_pipeline.text_encoder=get_peft_model(self.sd_pipeline.text_encoder,text_encoder_config)
                self.sd_pipeline.text_encoder.print_trainable_parameters()
            else:
                self.sd_pipeline.text_encoder.requires_grad_(True)
        elif train_text_encoder_embeddings:
            self.sd_pipeline.text_encoder.requires_grad_(False)
            self.sd_pipeline.text_encoder.get_input_embeddings().requires_grad_(True)
        if train_unet:
            if self.use_lora:
                self.sd_pipeline.unet.requires_grad_(False)
                lora_config = LoraConfig(
                    r=4,
                    lora_alpha=16,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                self.sd_pipeline.unet.add_adapter(lora_config)
                self.sd_pipeline.unet.print_trainable_parameters()
                '''
                # To avoid accelerate unscaling problems in FP16.
                for param in self.sd_pipeline.unet.parameters():
                    # only upcast trainable parameters (LoRA) into fp32
                    if param.requires_grad:
                        param.data = param.to(torch.float32)'''
            else:
                self.sd_pipeline.unet.requires_grad_(True)

    def get_trainable_layers(self):
        return [p for p in self.sd_pipeline.unet.parameters() if p.require_grad]+[p for p in self.sd_pipeline.text_encoder.parameters() if p.require_grad]