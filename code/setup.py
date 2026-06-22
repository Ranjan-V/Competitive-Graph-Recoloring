from setuptools import find_packages, setup


setup(
    name="competitive-recoloring",
    version="0.1.0",
    description="Deterministic simulations for competitive graph recoloring.",
    author="Applied Mathematics Letters study",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.2",
        "numpy>=1.24,<1.28",
        "matplotlib>=3.8",
    ],
    entry_points={
        "console_scripts": [
            "competitive-recoloring=competitive_recoloring.cli:main",
        ],
    },
)
