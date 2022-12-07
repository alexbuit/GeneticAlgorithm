from setuptools import setup, find_packages

setup(
    name="genetic_algorithm",
    author="Adrian v Eik",
    version="0.1",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "matplotlib", "scipy"],
    tests_require=["pytest"]
)