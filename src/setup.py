from setuptools import setup, find_packages

setup(
    name="ai4cop_health_cams",
    version="1.0",
    description="AI4Copernicus health bootstrapping service: Super-resolution of CAMS model output using deep learning",
    author="ECMWF",
    author_email="mihai.alexe@ecmwf.int",
    url="https://github.com/mishooax/ai4cop-health-cams",
    packages=find_packages(include=["aqgan", "ai4cop_health_cams.*"]),
    entry_points={
        "console_scripts": [
            "ai4cop-cams-train=ai4cop_health_cams.train:main",
            "ai4cop-cams-pretrain=ai4cop_health_cams.pretrain:main",
            "ai4cop-cams-predict=ai4cop_health_cams.predict:main",
        ]
    },
)
