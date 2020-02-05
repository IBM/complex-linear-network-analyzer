# +-----------------------------------------------------------------------------+
# |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                                       |
# |                                                                             |
# |  Licensed under the Apache License, Version 2.0 (the "License");            |
# |  you may not use this file except in compliance with the License.           |
# |  You may obtain a copy of the License at                                    |
# |                                                                             |
# |      http://www.apache.org/licenses/LICENSE-2.0                             |
# |                                                                             |
# |  Unless required by applicable law or agreed to in writing, software        |
# |  distributed under the License is distributed on an "AS IS" BASIS,          |
# |  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   |
# |  See the License for the specific language governing permissions and        |
# |  limitations under the License.                                             |
# +-----------------------------------------------------------------------------+
# |  Authors: Lorenz K. Mueller, Pascal Stark                                   |
# +-----------------------------------------------------------------------------+

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="complex-linear-network-analyzer",
    version="1.0.1",
    author="Lorenz K. Müller, Pascal Stark",
    author_email="crk@zurich.ibm.com",
    description="Computes analytically the output of complex valued, linear networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/complex-linear-network-analyzer",
    packages=['colna'],
    install_requires=['tqdm','scipy','numpy','matplotlib'],
    extras_require={
        'Visualization':["graphviz"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],

)
