#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <optional>

namespace py = pybind11;
namespace cbt = ::google::cloud::bigtable;

py::list SampleRowKey(py::object const& client, std::string const& table_id,
                      const std::optional<std::string>& app_profile_id) {
  auto project_id = client.attr("project_id").cast<std::string>();
  auto instance_id = client.attr("instance_id").cast<std::string>();

  if (app_profile_id) throw std::runtime_error("app_profile_id not supported.");

  auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                      cbt::ClientOptions());

  cbt::Table table(bigtable_client, table_id);

  auto samples = table.SampleRows();

  py::list res;

  for (auto const& resp : samples.value()) {
    std::string rk = resp.row_key;
    auto tuple = py::make_tuple(rk, resp.offset_bytes);
    res.append(tuple);
  }

  return res;
}

std::string ValueToBytes(float v) {
  // TODO(kboroszko): make it work like HBase
  return std::to_string(v);
}

float BytesToValue(std::string const& s) {
  // TODO(kboroszko): make it work like HBase
  return std::stof(s);
}

void putCellValueInTensor(torch::Tensor& tensor, int index, torch::Dtype dtype,
                          cbt::Cell const& cell) {
  switch (dtype) {
    case torch::kFloat32:
      tensor.index_put_({index}, BytesToValue(cell.value()));
      break;

    default:
      throw std::runtime_error("type not implemented");
  }
}

void Write(py::object const& client, std::string const& tableId,
           const std::optional<std::string>& appProfileId,
           torch::Tensor const& tensor, py::list const& columns,
           py::list const& row) {
  if (appProfileId) throw std::runtime_error("app_profile_id not supported.");
  auto project_id = client.attr("project_id").cast<std::string>();
  auto instance_id = client.attr("instance_id").cast<std::string>();

  auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                      cbt::ClientOptions());

  cbt::Table table(bigtable_client, tableId);

  auto* ptr = static_cast<float*>(tensor.data_ptr());

  for (int i = 0; i < tensor.size(0); i++) {
    auto row_key = row[i].cast<std::string>();

    for (int j = 0; j < tensor.size(1); j++) {
      auto col_name_full = columns[j].cast<std::string>();
      std::string col_family = col_name_full.substr(0, col_name_full.find(':'));
      std::string col_name = col_name_full.substr(col_name_full.find(':') + 1,
                                                  col_name_full.length());
      google::cloud::Status status = table.Apply(cbt::SingleRowMutation(
          row_key, cbt::SetCell(std::move(col_family), std::move(col_name),
                                ValueToBytes(*ptr))));
      ptr++;
      if (!status.ok()) throw std::runtime_error(status.message());
    }
  }
}

class BigtableTableIterator {
 public:
  BigtableTableIterator(cbt::RowReader reader,
                        const std::map<std::string, size_t>& column_map,
                        torch::Dtype dtype, const cbt::Table& table,
                        const std::shared_ptr<cbt::DataClient>& client)
      : reader_(std::move(reader)),
        column_map_(std::move(column_map)),
        dtype_(std::move(dtype)),
        table_(std::move(table)),
        client_(std::move(client)) {
    this->it_ = this->reader_.begin();
  }

  torch::Tensor next() {
    if (it_ == reader_.end()) throw py::stop_iteration();

    torch::Tensor tensor = torch::empty(this->column_map_.size(),
                                        torch::TensorOptions().dtype(dtype_));
    auto const& row = *it_;
    for (const auto& cell : row.value().cells()) {
      std::string full_name =
          cell.family_name() + ":" + cell.column_qualifier();
      putCellValueInTensor(tensor, column_map_[full_name], dtype_, cell);
    }

    it_ = std::next(it_);
    return tensor;
  }

 private:
  torch::Dtype dtype_;
  cbt::RowReader reader_;
  std::map<std::string, size_t> column_map_;
  cbt::v1::internal::RowReaderIterator it_;
  cbt::Table table_;
  std::shared_ptr<cbt::DataClient> client_;
};

cbt::Filter createColumnsFilter(const py::list& selected_columns) {
  std::vector<cbt::Filter> filters;

  for (const auto& column : selected_columns) {
    auto col_name_full = column.cast<std::string>();
    std::string col_family = col_name_full.substr(0, col_name_full.find(':'));
    std::string col_name = col_name_full.substr(col_name_full.find(':') + 1,
                                                col_name_full.length());
    cbt::Filter f =
        cbt::Filter::ColumnName(std::move(col_family), std::move(col_name));
    filters.push_back(std::move(f));
  }

  return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
}

BigtableTableIterator* CreateIterator(
    py::object const& client, std::string const& table_id,
    const std::optional<std::string>& app_profile_id, py::list const& samples,
    py::list const& selected_columns, std::string const& start_key,
    std::string const& end_key, std::string const& rowKey_prefix,
    std::string const& versions, const int num_workers, const int worker_id) {
  if (app_profile_id) throw std::runtime_error("app_profile_id not supported.");
  if (rowKey_prefix.length() >= 1)
    throw std::runtime_error("rowKey_prefix not supported.");
  if (versions != "latest") throw std::runtime_error("versions not supported.");

  std::cout << "got " << samples.size() << " samples.\n";

  std::cout << "running worker no. " << worker_id << " of " << num_workers
            << " in total.\n";
  auto project_id = client.attr("project_id").cast<std::string>();
  auto instance_id = client.attr("instance_id").cast<std::string>();

  std::map<std::string, size_t> column_map;
  size_t index = 0;
  for (const auto& column_name : selected_columns) {
    column_map[column_name.cast<std::string>()] = index++;
  }

  auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                      cbt::ClientOptions());

  cbt::Table table(bigtable_client, table_id);

  cbt::Filter filter_columns = createColumnsFilter(selected_columns);
  cbt::Filter filter =
      cbt::Filter::Chain(std::move(filter_columns), cbt::Filter::Latest(1));

  auto reader = table.ReadRows(
      cbt::RowRange::Range(std::move(start_key), std::move(end_key)),
      std::move(filter));

  return new BigtableTableIterator(std::move(reader), std::move(column_map),
                                   torch::kFloat32, std::move(table),
                                   std::move(bigtable_client));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sampleRowKey", &SampleRowKey, "get sample row_keys from BigTable",
        py::arg("client"), py::arg("table_id"),
        py::arg("application_profile_id") = py::none());

  m.def("write", &Write, "write tensor to BigTable", py::arg("client"),
        py::arg("table_id"), py::arg("app_profile_id") = py::none(),
        py::arg("tensor"), py::arg("columns"), py::arg("row_keys"));

  m.def("createIterator", &CreateIterator, "get BigTable ReadRows iterator",
        py::arg("client"), py::arg("table_id"),
        py::arg("app_profile_id") = py::none(), py::arg("samples"),
        py::arg("selected_columns"), py::arg("start_key"), py::arg("end_key"),
        py::arg("row_key_prefix"), py::arg("versions"), py::arg("num_workers"),
        py::arg("worker_id")

  );

  py::class_<BigtableTableIterator>(m, "Iterator")
      .def("__iter__",
           [](BigtableTableIterator& it) -> BigtableTableIterator& {
             return it;
           })
      .def("__next__", &BigtableTableIterator::next);
}