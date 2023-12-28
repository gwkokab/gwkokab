# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

_current_version = '0.0.2'

with open('README.md', encoding='utf-8') as f:
    _long_description = f.read()

keywords = [
    'jax', 'astronomy', 'astrophysics', 'machine-learning', 'deep-learning', 'bayesian-inference',
    'probabilistic-programming'
]

setup(
    name='jaxtro',
    version=_current_version,
    packages=find_packages(exclude=[
        'tests',
        'tests.*',
        'examples',
        'examples.*',
    ]),
    url='https://github.com/Qazalbash/jaxtro',
    license='Apache 2.0',
    author='Meesum Qazalbash and Muhammad Zeeshan',
    author_email='meesumqazalbash@gmail.com',
    maintainer='Meesum Qazalbash',
    maintainer_email='meesumqazalbash@gmail.com',
    description='A JAX-based gravitational-wave population inference',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    install_requires=['jaxampler', 'numpy', 'tqdm', 'configargparse'],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
    keywords=keywords,
    entry_points={"console_scripts": ["jaxtro_genie=jaxtro.main:main"]},
)
