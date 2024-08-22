from setuptools import setup

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7']

setup(name='microppo',
      description='MicroPPO - A PPO-based approach for managing power flows in micro-grid.',
      url='https://github.com/EbiDa/MicroPPO',
      version="0.0.1",
      license='AGPL-3.0-or-later',
      author='Daniel Ebi',
      classifiers=CLASSIFIERS,
      python_requires=">=3.7",
      packages=["microppo"],
      install_requires=["cvxpy", "cvxpylayers", "gym", "numpy", "pandas", "Requests", "scikit_learn",
                        "stable_baselines3==1.7.0", "torch", "tqdm"],
      extras_require={
          'plots': ["matplotlib>=2.0.0", "seaborn"]
      }
      )
