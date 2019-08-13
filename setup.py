from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='pysoc',
      version='0.0.1',
      description="Economic Self-Optimizing Control as python toolbox.",
      author="Victor Manuel Cunha Alves, Felipe Souza Lima",
      author_email='feslima93@gmail.com',
      url='https://github.com/feslima/pySOC',
      license='Apache License 2.0',
      packages=find_packages(exclude=['tests']),
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords="self-optimizing-control, plantwide-control",
      setup_requires=['numpy>=1.16'],
      install_requires=['scipy>=1.3.0'],
      python_requires='>=3.5',
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering"
      ]
      )
