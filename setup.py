#Responsible for creating my machine-learning application as a package
from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of the requirements
    """
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        #to remove the -e . while reading the file
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

#-e.  will automatically trigger the setup.py file

setup(
    name='Student-Performance-Prediction',
    version='0.0.1',
    author='umaisdev27',
    author_email='umaismldev@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
