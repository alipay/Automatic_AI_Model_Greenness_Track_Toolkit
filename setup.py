from setuptools import setup, find_packages
from setuptools.command.egg_info import egg_info


class egg_info_ex(egg_info):
    """Includes license file into `.egg-info` folder."""

    def run(self):
        # don't duplicate license into `.egg-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            # `install` command is in progress, copy license
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)

        egg_info.run(self)


VERSION = "1.3.11"

INSTALL_REQUIRES = [
    'psutil',
    'py-cpuinfo',
    'pynvml',
    'urllib3',
    'grpcio',
    'protobuf==3.19.6',
    'flask',
    'dataclasses'
]

if __name__ == '__main__':
    setup(
        name="greenness_track_toolkit",
        version=VERSION,
        license=("LICENSE"),
        url="https://www.alipay.com",
        author="GreenAI-OpenSource",
        author_email="antopenai@service.alipay.com",
        packages=find_packages(),
        include_package_data=True,
        platforms="linux3",
        install_requires=INSTALL_REQUIRES,
        keywords="GreenFlow",
        python_requires=">=3",
        entry_points={
            'console_scripts': [
                'greenness_track_toolkit = greenness_track_toolkit.__main__:main',
            ]},
        cmdclass={'egg_info': egg_info_ex},)