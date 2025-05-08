from setuptools import setup, find_packages

setup(
    name="vehicle_perception",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "ultralytics>=8.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "detectron2": [],  # Detectron2 requires manual installation
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
    },
    python_requires=">=3.8",
    description="Advanced Vehicle Perception System for autonomous vehicles",
    author="Bhavisha Chaudhari, Adarsh Jha",
    author_email="bhavisha2705@gmail.com, jhaadarsh350@gmail.com",
    url="https://github.com/Bhavisha-06/PATHvision",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
