from setuptools import find_packages, setup # type: ignore
from typing import List

HYPHEN_E="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)
    return requirements
setup(
    name='bengaluru-house-price', 
    version='0.0.1', 
    author='Shreedevi', 
    author_email='shreedevibalaji2004@gmail.com', 
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )