import pbt_C as code

def get_data_py(project_id, instance_id, table_id):
    print("example function in python calling c++ code")
    code.get_data(project_id, instance_id, table_id)