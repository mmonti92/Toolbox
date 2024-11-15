from setuptools import setup, find_packages

setup(
    name="DataAnalysis",
    version="0.0",
    install_requires=["numpy", "matplotlib", "scipy", "lmfit", "sympy"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/mmonti92/Toolbox",
    license="MIT",
    author="Maurizio Monti",
    author_email="monti.maurizi@gmail.com",
    description="Some data analysis code",
)
