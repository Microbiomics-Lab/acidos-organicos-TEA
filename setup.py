from setuptools import find_packages, setup

setup(
    name='acidos_organicos_tea',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Análisis de biomarcadores para detección de autismo usando Machine Learning',
    author='Esteven Aragón',
    license='MIT',
)