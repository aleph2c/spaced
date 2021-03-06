import setuptools

setuptools.setup(
  setup_requires=['pytest-runner'],
  tests_require=['pytest'],
  name='spaced',
  version=1.0,
  py_modules=['spaced'],
  install_requires=[
    'Click',
    'pyyaml',
    'pytest',
    'pyreadline',
    'numpy',
    'scipy',
    'matplotlib',
  ],
  entry_points='''
    [console_scripts]
    space=cli:cli
  ''',
)
