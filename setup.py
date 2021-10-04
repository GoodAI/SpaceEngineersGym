from setuptools import setup

setup(
    name="gym_space_engineers",
    version="0.0.9",
    install_requires=["gym", "numpy", "pyzmq"],
    extras_require={
        "tests": [
            # Test and env checker
            "pytest",
            "stable-baselines3",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
            # optional: reformat only changed lines
            # "darker",
        ],
    },
)
