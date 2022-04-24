import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="ECToolKits",
    version="0.1.2",
    author="Yongbin Zhuang",
    author_email="robinzhuang@outlook.com",
    description="Small Package to Postprocessing Results",
#    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.19.5",
        "matplotlib >= 3.3.2",
        "ase >= 3.20.1",
        "cp2kdata >= 0.2.0",
        "pandas"
  ],
    entry_points={
        'console_scripts': [
            'tlk=toolkit.main:main']
        }
)
