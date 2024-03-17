import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ovsd_tools",
    version="1.0.2",
    author="suk-6",
    author_email="me@suk.kr",
    description="Packages for simple implementations of Stable Diffusion with OpenVINO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suk-6/ovsd-tools",
    packages=setuptools.find_packages(include=["ovsd_tools", "ovsd_tools.*"]),
    install_requires=[
        "torch",
        "openvino",
        "diffusers",
        "transformers",
        "accelerate",
        "scipy",
        "huggingface_hub",
        "opencv-python",
    ],
    python_requires=">=3.6",
    license="MIT",
)
