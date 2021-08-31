import pbt_C as code

def get_data_py(project_id, instance_id, table_id):
    print("example function in python calling c++ code")
    code.get_data(project_id, instance_id, table_id)

def main():
    get_data_py('fake_project', 'fake_instance', 'fake_table')

if __name__ == "__main__":
    # execute only if run as a script
    main()
