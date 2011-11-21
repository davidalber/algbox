from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    'matplotlib >= 1.0.1',
    'numpy >= 1.6.1'
]


setup(name='algbox',
    version=version,
    description="A collection of implmentations done for learning.",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='',
    author='David Alber',
    author_email='alber.david@gmail.com',
    url='http://www.davidalber.net/',
    license='MIT',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points={
        'console_scripts':
            ['conhull=algbox:convex_hulls.convex_hulls',
             'delaunay=algbox:delaunay.delaunay']
    },
    test_suite = 'tests.tests'
)
