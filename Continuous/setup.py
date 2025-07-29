import setuptools

setuptools.setup(
    name="geometric_plan_c",
    version="0.1",
    author="Anonymous",
    description="Return all solutions of a geometric problem",
    ext_modules=[
        setuptools.Extension(
            "geometric_plan_c",
            ["geometric_plan.cpp"],
            language="c++",
            include_dirs=["/usr/include/eigen3", "/usr/local/include/ompl-1.6"]
            # /usr/local/include/ompl-1.6
        )
    ],
    install_requires=[
        'pybind11>=2.7',
    ],
)