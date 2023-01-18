from setuptools import setup, find_packages

setup(
    name="dfmcontrol",
    packages=find_packages(),
    author="Adrian v Eik",
    version="0.1",
    license="MIT",
    install_requires=["numpy", "pandas", "matplotlib", "scipy"],
    tests_require=["pytest"]
)
