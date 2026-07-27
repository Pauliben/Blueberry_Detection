from setuptools import setup

setup(
    name='FPTeam',
    version='0.1.0',    
    description='FPTeam tools for CV',
    url='',
    author='Bruno Leme',
    author_email='b.caixetaleme@ufl.edu',
    license='BSD 2-clause',
    packages=['FPTeam'],
    install_requires=['numpy',     
                      'sahi', 
                      'ultralytics'                
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)