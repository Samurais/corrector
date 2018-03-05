# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
LONGDOC = """
Synonyms
=====================

Chinese Text Error Correction for Natural Language Processing and Understanding.

Welcome
-------

中文错别字纠正工具。音似、形似错字（或变体字）纠正，可用于中文拼音、笔画输入法的错误纠正。python开发。

"""

setup(
    name='error-correction',
    version='1.0.3',
    description='Chinese Text Error Correction for Natural Language Processing and Understanding',
    long_description=LONGDOC,
    author='Hai Liang Wang',
    author_email='hailiang.hl.wang@gmail.com',
    url='https://github.com/Samurais/corrector',
    license="MIT",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic'],
    keywords='corpus,machine-learning,NLU,NLP,Synonyms,Similarity',
    packages=find_packages(),
    install_requires=[
        'kenlm==0.0.0',
        'numpy>=1.13.1',
        'absl-py==0.1.10',
        'pypinyin==0.30.0'
    ],
    package_data={
        'corrector': [
            'data/kenlm/*',
            '**/*.gz',
            '**/*.txt',
            '**/*.klm',
            '**/*.arpa',
            '**/*.pkl',
            'LICENSE']})
