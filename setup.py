from setuptools import setup, find_packages

setup(
    name="otora",
    version="0.1.0",
    description="OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents",
    author="OTora Authors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "accelerate>=0.26.0",
        "pandas",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "icl": ["openai"],
        "vllm": ["vllm"],
    },
)
