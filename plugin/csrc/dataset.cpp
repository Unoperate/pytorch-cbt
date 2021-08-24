#include <torch/extension.h>
#include <torch/torch.h>

#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"

#include <iostream>
#include <vector>

void get_data(std::string const project_id, std::string const instance_id, std::string const table_id)
{
  namespace cbt = ::google::cloud::bigtable;

  cbt::Table table(cbt::CreateDefaultDataClient(project_id, instance_id,
                                                cbt::ClientOptions()),
                   table_id);

  google::cloud::bigtable::v1::RowReader reader1 = table.ReadRows(
      cbt::RowRange::InfiniteRange(), cbt::Filter::PassAllFilter());

  for (auto const &row : reader1)
  {
    if (!row)
      throw std::runtime_error(row.status().message());
    std::cout << "row: " << row->row_key() << ":\n";
    for (auto const &cell : row->cells())
    {
      std::cout << "cell:\n";
      std::cout << cell.family_name() << ":" << cell.column_qualifier() << ":"
                << cell.value() << "       @ " << cell.timestamp().count()
                << "us\n";
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("get_data", &get_data, "get data from BigTable");
}
