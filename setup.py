from setuptools import setup

setup(
    name='coheating-analysis',
    version='0.1.0',
    description='Tools for data analysis of a coheating test',
    url='https://github.com/Sarah-Ju/coheating-analysis',
    author='Sarah Juricic',
    author_email='sarah.juricic@gmail.fr',
    license='MIT',
    packages=['coheating'],
    install_requires=['numpy>=1.25.0',
                      'pandas>=1.5.3',
                      'statsmodels>=0.13.5'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License:: OSI Approved:: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
    ],
)
