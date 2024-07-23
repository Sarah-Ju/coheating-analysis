from setuptools import setup, find_packages

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()

setup(
    name='coheating-analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    author='Sarah Juricic',
    author_email='sarah.juricic@gmail.com',
    description='Tools for Heat Transfer Coefficient estimation from a coheating test',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Sarah-Ju/coheating-analysis',
    license='MIT',
)
