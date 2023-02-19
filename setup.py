import setuptools
packages = setuptools.find_packages()
print(f"Found the foullowing packages: {packages}")
setuptools.setup(name='Adversarial-Robustness',
                 packages=packages)
