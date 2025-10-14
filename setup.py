from setuptools import setup, find_packages

setup(
    name="moffitt_agentic_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.1",
        "langchain_community>=0.0.16",
        "langchain-chroma>=0.1.0",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.22",
        "pydantic>=2.5.2",
        "ollama>=0.1.5",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.29.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "moffitt-rag-app=moffitt_rag.streamlit.app:main",
        ],
    },
    author="Moffitt Cancer Center Team",
    author_email="example@moffitt.org",
    description="Agentic RAG System for Moffitt Cancer Center Researcher Data",
    keywords="rag, langchain, research, cancer",
    python_requires=">=3.10",
)