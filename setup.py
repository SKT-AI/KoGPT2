from kogpt2 import __version__
from setuptools import find_packages, setup

setup(name='kogpt2',
      version=__version__,
      url='https://github.com/SKT-AI/KoGPT2',
      license='midified MIT',
      author='Heewon Jeon',
      author_email='madjakarta@gmail.com',
      description='KoGPT2 (Korean GPT-2)',
      packages=find_packages(where=".", exclude=(
          'tests',
          'scripts'
      )),
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      )
