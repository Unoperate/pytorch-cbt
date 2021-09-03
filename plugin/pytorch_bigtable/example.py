import torch
import pbt_C


class client():
    def __init__(self) -> None:
        self.project_id = "fake_project"
        self.instance_id = "fake_instance"


def main():
    ten = torch.Tensor([x for x in range(40)]).reshape(20, 2)
    pbt_C.write(client(),  # client
                "fake_table",  # table_id
                None,  # app_profile_id
                ten,  # tensor
                ["cf1:c1", "cf1:c2"],  # columns
                ["row" + str(i).rjust(3, '0') for i in range(20)])  # row_keys

    it = pbt_C.createIterator(client(),
                              "fake_table",  # table_id
                              None,  # app_profile_id
                              ["s1"],  # samples
                              ["cf1:c2", "cf1:c1"],  # selected_columns
                              "row000",  # start_keyu
                              "row025",  # end_key
                              "",  # prefix
                              "latest",  # versions
                              1,  # numworkers
                              0, )  # worker id

    for x in it:
        print(x)


if __name__ == "__main__":
    # execute only if run as a script
    main()
