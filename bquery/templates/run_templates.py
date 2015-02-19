from jinja2 import Environment, PackageLoader
env = Environment(loader=PackageLoader('bquery', 'templates'))

template = env.get_template('ctable_ext.template.pyx')

with open('../ctable_ext.pyx', mode='w') as fout:
    fout.write(template.render())