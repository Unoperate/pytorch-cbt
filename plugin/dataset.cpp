#include <torch/extension.h>

#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include "google/cloud/bigtable/rpc_retry_policy.h"

#include <iostream>
#include <vector>

std::vector<at::Tensor> test_func(torch::Tensor input, int x) {
  return {input * x, input};
}

std::string get_data(
    std::string project_id,
    std::string instance_id,
    std::string table_id,
    std::string column_family,
    std::string column_name) {
  std::string x = project_id + instance_id + "hello_world";
  return x;
}

void foo() {
	google::cloud::bigtable::DefaultRPCRetryPolicy(google::cloud::bigtable::internal::kBigtableInstanceAdminLimits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_func", &test_func, "test function");
  m.def("get_data", &get_data, "get data from BigTable");
}
