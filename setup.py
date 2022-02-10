from distutils.core import setup
setup(
    name='dlp',
    packages=['dlp'],
    version='1.0.76',
    license='MIT', # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Code generator',
    author='BAO',
    author_email='hqbao1991@gmail.com',
    url='https://github.com/hqbao',
    download_url='https://github.com/hqbao/dlp_codegen/archive/v1.0.0.tar.gz',
    keywords=['deeplearning', 'computervision', 'deeplearningplatform', 'ai'],
    install_requires=[
        'tensorflow',
        'numpy',
        'scikit-image',
        'scipy',
        'kaggle',
        'requests',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_data={'dlp': ['code_templates/*']},
)