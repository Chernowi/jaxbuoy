from setuptools import setup


package_name = "buoy_agent_inference"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/inference.launch.py"]),
        (f"share/{package_name}/config", ["config/inference_params.yaml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="pedro",
    maintainer_email="pedro@example.com",
    description="Minimal NumPy inference node for buoy RNN policy",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = buoy_agent_inference.inference_node:main",
        ],
    },
)
