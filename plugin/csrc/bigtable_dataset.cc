#include <google/cloud/bigtable/table.h>
#include <google/cloud/bigtable/table_admin.h>
#include <google/protobuf/text_format.h>
#include <pybind11/pybind11.h>
#include <rpc/rpc.h> /* xdr is a sub-library of rpc */
#include <torch/extension.h>
#include <torch/torch.h>
#include <optional>

namespace py = pybind11;
namespace cbt = ::google::cloud::bigtable;

namespace {

std::string FloatToBytes(float v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_float(&xdrs, &v)) {
    throw std::runtime_error("Error writing float to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

float BytesToFloat(std::string s) {
  float v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_float(&xdrs, &v)) {
    throw std::runtime_error("Error reading float from byte array.");
  }
  return v;
}

void putCellValueInTensor(torch::Tensor& tensor, int index, torch::Dtype dtype,
                          cbt::Cell const& cell) {
  switch (dtype) {
    case torch::kFloat32:
      tensor.index_put_({index}, BytesToFloat(cell.value()));
      break;

    default:
      throw std::runtime_error("type not implemented");
  }
}

cbt::Filter createColumnsFilter(
    const std::map<std::pair<std::string, std::string>, size_t>&
        selected_columns) {
  std::vector<cbt::Filter> filters;

  for (const auto& key : selected_columns) {
    std::pair<std::string, std::string> pair = key.first;
    cbt::Filter f = cbt::Filter::ColumnName(pair.first, pair.second);
    filters.push_back(std::move(f));
  }

  return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
}

std::pair<std::string, std::string> ColumnNameToPair(
    const std::string& col_name_full) {
  size_t delimiter_pos = col_name_full.find(':');
  if (delimiter_pos == std::string::npos)
    throw std::invalid_argument("Invalid column name:" + col_name_full +
                                "\nColumn name must be in format " +
                                "column_family:column_name.");
  std::string col_family = col_name_full.substr(0, delimiter_pos);
  std::string col_name =
      col_name_full.substr(delimiter_pos + 1, col_name_full.length());
  std::pair<std::string, std::string> pair(col_family, col_name);
  return pair;
}

}  // namespace

py::list SampleRowKey(py::object const& client, std::string const& table_id,
                      const std::optional<std::string>& app_profile_id) {
  auto project_id = client.attr("project_id").cast<std::string>();
  auto instance_id = client.attr("instance_id").cast<std::string>();

  if (app_profile_id) throw std::runtime_error("app_profile_id not supported.");

  auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                      cbt::ClientOptions());

  cbt::Table table(bigtable_client, table_id);

  google::cloud::StatusOr<std::vector<cbt::RowKeySample>> samples =
      table.SampleRows();
  if (!samples.ok()) throw std::runtime_error(samples.status().message());

  py::list res;

  for (auto const& resp : samples.value()) {
    std::string rk = resp.row_key;
    auto tuple = py::make_tuple(rk, resp.offset_bytes);
    res.append(tuple);
  }

  return res;
}

void Write(py::object const& client, std::string const& tableId,
           const std::optional<std::string>& appProfileId,
           torch::Tensor const& tensor, py::list const& columns,
           py::list const& row) {
  if (appProfileId)
    throw std::invalid_argument("app_profile_id not supported.");
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
                                FloatToBytes(*ptr))));
      ++ptr;
      if (!status.ok()) throw std::runtime_error(status.message());
    }
  }
}

class BigtableDatasetIterator {
 public:
  BigtableDatasetIterator(
      cbt::RowReader reader,
      std::map<std::pair<std::string, std::string>, size_t> const& column_map,
      torch::Dtype dtype, cbt::Table const& table,
      std::shared_ptr<cbt::DataClient> const& client)
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
      std::pair<std::string, std::string> key(cell.family_name(),
                                              cell.column_qualifier());
      putCellValueInTensor(tensor, column_map_[key], dtype_, cell);
    }

    it_ = std::next(it_);
    return tensor;
  }

 private:
  torch::Dtype dtype_;
  cbt::RowReader reader_;
  std::map<std::pair<std::string, std::string>, size_t> column_map_;
  cbt::v1::internal::RowReaderIterator it_;
  cbt::Table table_;
  std::shared_ptr<cbt::DataClient> client_;
};

BigtableDatasetIterator* CreateIterator(
    py::object const& client, std::string const& table_id,
    std::optional<std::string> const& app_profile_id, py::list const& samples,
    py::list const& selected_columns, std::string const& start_key,
    std::string const& end_key, std::string const& rowKey_prefix,
    std::string const& versions, const int num_workers, const int worker_id) {
  if (app_profile_id)
    throw std::invalid_argument("app_profile_id not supported.");
  if (rowKey_prefix.length() >= 1)
    throw std::invalid_argument("rowKey_prefix not supported.");
  if (versions != "latest")
    throw std::invalid_argument("only `version`='latest' is supported.");

  std::cout << "got " << samples.size() << " samples.\n";

  std::cout << "running worker no. " << worker_id << " of " << num_workers
            << " in total.\n";
  auto const project_id = client.attr("project_id").cast<std::string>();
  auto const instance_id = client.attr("instance_id").cast<std::string>();

  // we're using a regular map because unordered_map cannot hash a pair by
  // default.
  std::map<std::pair<std::string, std::string>, size_t> column_map;
  size_t index = 0;
  for (const auto& column_name : selected_columns) {
    std::pair<std::string, std::string> pair =
        ColumnNameToPair(column_name.cast<std::string>());
    column_map[pair] = index++;
  }

  auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                      cbt::ClientOptions());

  cbt::Table table(bigtable_client, table_id);

  cbt::Filter filter_columns = createColumnsFilter(column_map);
  cbt::Filter filter =
      cbt::Filter::Chain(std::move(filter_columns), cbt::Filter::Latest(1));

  auto reader = table.ReadRows(
      cbt::RowRange::Range(std::move(start_key), std::move(end_key)),
      std::move(filter));

  return new BigtableDatasetIterator(std::move(reader), std::move(column_map),
                                     torch::kFloat32, std::move(table),
                                     std::move(bigtable_client));
}

std::string PrintRowRange(cbt::RowRange const& row_range) {
  std::string res;
  google::protobuf::TextFormat::PrintToString(row_range.as_proto(), &res);
  return res;
}

std::string PrintRowSet(cbt::RowSet const& row_set) {
  std::string res;
  google::protobuf::TextFormat::PrintToString(row_set.as_proto(), &res);
  return res;
}

void AppendRowOrRange(cbt::RowSet& row_set, py::args const& args) {
  for (auto const& arg : args) {
    if (py::isinstance<cbt::RowRange>(arg))
      row_set.Append(arg.cast<cbt::RowRange>());
    else if (py::isinstance<py::str>(arg))
      row_set.Append(arg.cast<std::string>());
    else
      throw py::type_error(
          "argument must be a row (str) or a range (RowRange)");
  }
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

  py::class_<BigtableDatasetIterator>(m, "Iterator")
      .def("__iter__",
           [](BigtableDatasetIterator& it) -> BigtableDatasetIterator& {
             return it;
           })
      .def("__next__", &BigtableDatasetIterator::next);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<cbt::RowRange>(m, "RowRange").def("__repr__", &PrintRowRange);

  m.def("infinite_row_range", &cbt::RowRange::InfiniteRange,
        "Create an infinite row range");
  m.def("starting_at_row_range", &cbt::RowRange::StartingAt<std::string>,
        "Create a row range from given row key to infinity",
        py::arg("row_key"));
  m.def("ending_at_row_range", &cbt::RowRange::EndingAt<std::string>,
        "Create a row range from infinity to given row key",
        py::arg("row_key"));
  m.def("empty_row_range", &cbt::RowRange::Empty, "Create an empty row range");
  m.def("prefix_row_range", &cbt::RowRange::Prefix<std::string>,
        "Create a row range of rows starting with given prefix",
        py::arg("prefix"));
  m.def("right_open_row_range",
        &cbt::RowRange::RightOpen<std::string, std::string>,
        "Create a row range with start inclusive and end exclusive",
        py::arg("start"), py::arg("end"));
  m.def("left_open_row_range",
        &cbt::RowRange::LeftOpen<std::string, std::string>,
        "Create a row range with start exclusive and end inclusive",
        py::arg("start"), py::arg("end"));
  m.def("open_row_range", &cbt::RowRange::Open<std::string, std::string>,
        "Create a row range with start and end both exclusive",
        py::arg("start"), py::arg("end"));
  m.def("closed_row_range", &cbt::RowRange::Closed<std::string, std::string>,
        "Create a row range with start and end both inclusive",
        py::arg("start"), py::arg("end"));

  py::class_<cbt::RowSet>(m, "RowSet")
      .def(py::init<>())
      .def("append_row", &cbt::RowSet::Append<std::string>, py::arg("row_key"))
      .def("append_range",
           static_cast<void (cbt::RowSet::*)(cbt::RowRange)>(
               &cbt::RowSet::Append),
           py::arg("row_range"))
      .def("append", &AppendRowOrRange)
      .def("intersect", &cbt::RowSet::Intersect, py::arg("row_range"))
      .def("__repr__", &PrintRowSet);
}
