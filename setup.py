import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ComplexLinearNetworkAnalyzer",
    version="0.0.1",
    author="Lorenz K. Müller, Pascal Stark",
    author_email="crk@zurich.ibm.com",
    description=" Computes analytic output of complex valued, linear networks.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.ibm.com/lmu-zurich/analyticalReservoir",
    packages=['colna'],
    install_requires=['tqdm','scipy','numpy','matplotlib'],
    extras_require={
        'Visualization':["graphviz"]
    }
)
