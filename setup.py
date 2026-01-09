from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="slurm-monitor",
    version="0.1.0",
    author="Dongwoo Kim",
    author_email="dongwookim.ml@gmail.com",
    description="A real-time terminal dashboard for monitoring SLURM cluster jobs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dongwookim-ml/slurm-monitor",
    py_modules=["slurm_monitor"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Monitoring",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        "rich>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "slurm-monitor=slurm_monitor:main",
        ],
    },
)
