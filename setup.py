from setuptools import find_packages , setup
requires = ['blinker==1.7.0', 'cachetools==5.3.2', 'certifi==2023.11.17', 'charset-normalizer==3.3.2', 'click==8.1.7', 'enum-compat==0.0.3', 'filelock==3.13.1', 'Flask==3.0.0', 'fsspec==2023.12.2', 'google-api-core==2.15.0', 'google-auth==2.26.2', 'google-cloud-core==2.4.1', 'google-cloud-storage==2.14.0', 'google-crc32c==1.5.0', 'google-resumable-media==2.7.0', 'googleapis-common-protos==1.62.0', 'h11==0.14.0', 'huggingface-hub==0.20.1', 'idna==3.6', 'itsdangerous==2.1.2', 'Jinja2==3.1.2', 'MarkupSafe==2.1.3', 'mpmath==1.3.0', 'networkx==3.2.1', 'numpy==1.26.2', 'packaging==23.2', 'pandas==2.1.4', 'pillow==10.2.0', 'protobuf==4.25.2', 'psutil==5.9.8', 'pyasn1==0.5.1', 'pyasn1-modules==0.3.0', 'python-dateutil==2.8.2', 'pytz==2023.3.post1', 'PyYAML==6.0.1', 'regex==2023.10.3', 'requests==2.31.0', 'rsa==4.9', 'safetensors==0.4.1', 'six==1.16.0', 'sympy==1.12', 'tokenizers==0.15.0', 'torch==2.1.2', 'torch-model-archiver==0.9.0', 'torchserve==0.9.0', 'tqdm==4.66.1', 'transformers==4.36.2', 'triton==2.1.0', 'typing_extensions==4.9.0', 'tzdata==2023.3', 'urllib3==2.1.0', 'uvicorn==0.25.0', 'Werkzeug==3.0.1']
setup(
    name='model-app',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    description='estiMate training package',
    package_data={'': ['*.csv','*.json']},
)
# test change