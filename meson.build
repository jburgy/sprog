project('sprog', 'cython')

py = import('python').find_installation(pure: false)
dep_py = py.dependency()

py.extension_module(
    'sprog',
    ['src/sprog.pyx'],
    dependencies: dep_py,
    install: true,
    include_directories: ['src'],
)