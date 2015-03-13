from jinja2 import  FileSystemLoader, Environment

templateLoader = FileSystemLoader(searchpath="./bquery/templates/")
env = Environment(loader=templateLoader)

template = env.get_template('ctable_ext.template.pyx')

with open('bquery/ctable_ext.pyx', mode='w') as fout:
    fout.write(template.render(
        factor_types=['int64', 'int32', 'float64'],
        count_unique_types=['float64', 'int64', 'int32', ],
        sum_types=['float64', 'int32', 'int64'])
    )
# Run from bquery top folder run:
# python bquery/templates/run_templates.py