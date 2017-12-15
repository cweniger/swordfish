from setuptools import setup

setup(
    name='swordfish',
    version='0.2.0',
    description='Predicting the Information Yield of Counting Experiments.',
    author='Thomas Edwards and Christoph Weniger',
    author_email='c.weniger@uva.nl',
    url = 'https://github.com/cweniger/swordfish',
    download_url = 'https://github.com/cweniger/swordfish/archive/0.1.tar.gz',
    packages=['swordfish'],
    package_data={'swordfish': [] },
    long_description=open('README.md').read(),
)
