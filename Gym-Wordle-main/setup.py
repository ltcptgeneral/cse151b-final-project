from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='gym_wordle',
    version='0.1.3',
    author='David Kraemer',
    author_email='david.kraemer@stonybrook.edu',
    description='OpenAI gym environment for training agents on Wordle',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DavidNKraemer/Gym-Wordle',
    packages=find_packages(
        include=[
            'gym_wordle',
            'gym_wordle.*'
        ]
    ),
    package_data={
        'gym_wordle': ['dictionary/*']
    },
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.20',
        'gym==0.19',
        'sty==1.0',
    ],
)
