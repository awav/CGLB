# Copyright 2021 The CGLB Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

pkgs = find_packages()

setup(
    name="cglb",
    author="Artem Artemev, David Burt",
    author_email="a.artemev20@imperial.ac.uk, drb62@cam.ac.uk",
    version="0.0.1",
    packages=pkgs,
    install_requires=["numpy", "scipy"],
    dependency_links=[],
)
