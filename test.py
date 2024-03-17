import gradio as gr
from ovsd_tools import ovsd_tools

ovsd = ovsd_tools()
ovsd.huggingface_hub_login()
ovsd.createPytorchModelPipe()
ovsd.IRConversion()
ovsd.createOVModelPipe()

ov_pipe = ovsd.ov_pipe


def imageGeneration(prompt, seed, num_steps):
    result = ov_pipe(prompt, seed=seed, num_inference_steps=num_steps)
    return result["sample"][0]


demo = gr.Interface(
    fn=imageGeneration,
    inputs=[
        gr.Textbox(label="Prompt", value=ovsd.sample_prompt),
        gr.Slider(0, 10000000, 42, label="Seed"),
        gr.Slider(1, 50, 20, label="Num Steps"),
    ],
    outputs=[
        gr.Image(label="Generated Image", type="numpy", show_label=True),
    ],
)

demo.launch(share=True)
