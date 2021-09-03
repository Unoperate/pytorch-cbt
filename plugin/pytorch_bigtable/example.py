import torch
import pbt_C as code


class client():
    def __init__(self) -> None:
        self.project_id = "fake_project"
        self.instance_id = "fake_instance"


def main():
    ten = torch.Tensor([x for x in range(40)]).reshape(20, 2)
    pbt_C.io_big_table_write(client(),              # client
                             "fake_table",          # table_id
                             None,                  # app_profile_id
                             ten,                   # tensor
                             ["cf1:c1", "cf1:c2"],  # columns
                             ["row" + str(i).rjust(3, '0') for i in range(20)])  # row_keys


if __name__ == "__main__":
    # execute only if run as a script
    main()
