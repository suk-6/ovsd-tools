import gc
import openvino as ov
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline
from huggingface_hub import notebook_login, whoami, interpreter_login
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.schedulers import LMSDiscreteScheduler

from ovsd_tools.options import *
from ovsd_tools.utils.tools import *


class ovsd_tools:
    def __init__(self):
        self._device = "CPU"
        self._ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    def huggingface_hub_login(self):
        try:
            whoami()
            print("Authorization token already provided")
        except OSError:
            try:
                notebook_login()
            except ImportError:
                interpreter_login()

    def createPytorchModelPipe(self):
        pipe = StableDiffusionPipeline.from_pretrained("prompthero/openjourney").to(
            "cpu"
        )

        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()

        self.unet = pipe.unet
        self.unet.eval()

        self.vae = pipe.vae
        self.vae.eval()

        del pipe
        gc.collect()

    def IRConversion(self):
        if not TEXT_ENCODER_OV_PATH.exists():
            convert_encoder(self.text_encoder, TEXT_ENCODER_OV_PATH)
        else:
            print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

        del self.text_encoder
        gc.collect()

        if not UNET_OV_PATH.exists():
            convert_unet(self.unet, UNET_OV_PATH)
            gc.collect()
        else:
            print(f"Unet will be loaded from {UNET_OV_PATH}")
        del self.unet
        gc.collect()

        if not VAE_ENCODER_OV_PATH.exists():
            convert_vae_encoder(self.vae, VAE_ENCODER_OV_PATH)
        else:
            print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH}")

        if not VAE_DECODER_OV_PATH.exists():
            convert_vae_decoder(self.vae, VAE_DECODER_OV_PATH)
        else:
            print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

        del self.vae
        gc.collect()

    def createOVModelPipe(self):
        core = ov.Core()

        device = "CPU"

        text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, device)
        unet_model = core.compile_model(UNET_OV_PATH, device)

        ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device != "CPU" else {}

        vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device, ov_config)
        vae_encoder = core.compile_model(VAE_ENCODER_OV_PATH, device, ov_config)

        lms = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self._ov_pipe = OVStableDiffusionPipeline(
            tokenizer=tokenizer,
            text_encoder=text_enc,
            unet=unet_model,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            scheduler=lms,
        )

    @property
    def ov_pipe(self):
        return self._ov_pipe

    @property
    def sample_prompt(self):
        return (
            "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting, epic composition. "
            "A golden daylight, hyper-realistic environment. "
            "Hyper and intricate detail, photo-realistic. "
            "Cinematic and volumetric light. "
            "Epic concept art. "
            "Octane render and Unreal Engine, trending on artstation"
        )
