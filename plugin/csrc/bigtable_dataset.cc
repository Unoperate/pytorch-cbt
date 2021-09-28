#include <google/cloud/bigtable/table.h>
#include <google/cloud/bigtable/table_admin.h>
#include <google/protobuf/text_format.h>
#include <grpcpp/security/credentials.h>
#include <pybind11/numpy.h>
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

float BytesToFloat(std::string const& s) {
  float v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_float(&xdrs, &v)) {
    throw std::runtime_error("Error reading float from byte array.");
  }
  return v;
}

std::string DoubleToBytes(double v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_double(&xdrs, &v)) {
    throw std::runtime_error("Error writing double to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

double BytesToDouble(std::string const& s) {
  double v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_double(&xdrs, &v)) {
    throw std::runtime_error("Error reading double from byte array.");
  }
  return v;
}

std::string Int64ToBytes(int64_t v) {
  char buffer[sizeof(v)];
  XDR xdrs;
  xdrmem_create(&xdrs, buffer, sizeof(v), XDR_ENCODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    throw std::runtime_error("Error writing int64 to byte array.");
  }
  std::string s(buffer, sizeof(v));
  return s;
}

int64_t BytesToInt64(std::string const& s) {
  int64_t v;
  XDR xdrs;
  xdrmem_create(&xdrs, const_cast<char*>(s.data()), sizeof(v), XDR_DECODE);
  if (!xdr_int64_t(&xdrs, &v)) {
    throw std::runtime_error("Error reading int64 from byte array.");
  }
  return v;
}

void PutCellValueInTensor(torch::Tensor* tensor, int index,
                          torch::Dtype cell_type, cbt::Cell const& cell) {
  switch (cell_type) {
    case torch::kFloat32:
      tensor->index_put_({index}, BytesToFloat(cell.value()));
      break;
    case torch::kFloat64:
      tensor->index_put_({index}, BytesToDouble(cell.value()));
      break;
    case torch::kI64:
      tensor->index_put_({index}, BytesToInt64(cell.value()));
      break;
    default:
      throw std::runtime_error("type not implemented");
  }
}

std::string GetTensorValueAsBytes(torch::Tensor const& tensor, size_t i,
                                  size_t j) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32: {
      auto tensor_ptr = tensor.accessor<float, 2>();
      return FloatToBytes(tensor_ptr[i][j]);
    }
    case torch::kFloat64: {
      auto tensor_ptr = tensor.accessor<double, 2>();
      return DoubleToBytes(tensor_ptr[i][j]);
    }
    case torch::kInt64: {
      auto tensor_ptr = tensor.accessor<int64_t, 2>();
      return Int64ToBytes(tensor_ptr[i][j]);
    }
    default:
      throw std::runtime_error("type not implemented");
  }
}

cbt::Filter CreateColumnsFilter(
    std::map<std::pair<std::string, std::string>, size_t> const& columns) {
  std::vector<cbt::Filter> filters;

  for (const auto& key : columns) {
    std::pair<std::string, std::string> pair = key.first;
    cbt::Filter f = cbt::Filter::ColumnName(pair.first, pair.second);
    filters.push_back(std::move(f));
  }

  return cbt::Filter::InterleaveFromRange(filters.begin(), filters.end());
}

std::pair<std::string, std::string> ColumnNameToPair(
    std::string const& col_name_full) {
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

std::unique_ptr<cbt::Table> CreateTable(
    std::shared_ptr<cbt::DataClient> const& data_client,
    std::string const& table_id,
    std::optional<std::string> const& app_profile_id) {
  return app_profile_id ? std::make_unique<cbt::Table>(
                              data_client, *std::move(app_profile_id), table_id)
                        : std::make_unique<cbt::Table>(data_client, table_id);
}

py::list SampleRowKeys(std::shared_ptr<cbt::DataClient> const& data_client,
                       std::string const& table_id,
                       std::optional<std::string> const& app_profile_id) {
  auto table = CreateTable(data_client, table_id, app_profile_id);

  auto maybe_sample_row_keys = table->SampleRows();
  if (!maybe_sample_row_keys.ok())
    throw std::runtime_error(maybe_sample_row_keys.status().message());
  auto& sample_row_keys = maybe_sample_row_keys.value();

  py::list res;
  for (auto const& resp : sample_row_keys) {
    res.append(py::make_tuple(resp.row_key, resp.offset_bytes));
  }

  return res;
}

void WriteTensor(std::shared_ptr<cbt::DataClient> const& data_client,
                 std::string const& table_id,
                 std::optional<std::string> const& app_profile_id,
                 torch::Tensor const& tensor, py::list const& columns,
                 py::list const& row) {
  auto table = CreateTable(data_client, table_id, app_profile_id);

  for (int i = 0; i < tensor.size(0); i++) {
    auto row_key = row[i].cast<std::string>();

    for (int j = 0; j < tensor.size(1); j++) {
      auto col_name_full = columns[j].cast<std::string>();
      auto [col_family, col_name] = ColumnNameToPair(col_name_full);
      google::cloud::Status status = table->Apply(cbt::SingleRowMutation(
          row_key, cbt::SetCell(std::move(col_family), std::move(col_name),
                                GetTensorValueAsBytes(tensor, i, j))));
      if (!status.ok()) throw std::runtime_error(status.message());
    }
  }
}

torch::Tensor getFilledTensor(size_t size, torch::Dtype const& cell_type,
                              std::optional<py::object> const& default_value) {
  switch (cell_type) {
    case torch::kFloat32:
      return default_value
                 ? torch::full(size, (*default_value).cast<float>(),
                               torch::TensorOptions().dtype(cell_type))
                 : torch::full(size, NAN,
                               torch::TensorOptions().dtype(cell_type));
    case torch::kFloat64:
      return default_value
                 ? torch::full(size, (*default_value).cast<double>(),
                               torch::TensorOptions().dtype(cell_type))
                 : torch::full(size, NAN,
                               torch::TensorOptions().dtype(cell_type));
    case torch::kI64:
      return default_value
                 ? torch::full(size, (*default_value).cast<int64_t>(),
                               torch::TensorOptions().dtype(cell_type))
                 : torch::zeros(size, torch::TensorOptions().dtype(cell_type));
    default:
      throw std::runtime_error("type not implemented");
  }
}

class BigtableDatasetIterator {
 public:
  BigtableDatasetIterator(std::shared_ptr<cbt::DataClient> const& data_client,
                          std::string const& table_id,
                          std::optional<std::string> const& app_profile_id,
                          py::list const& /*sample_row_keys*/,
                          py::list const& columns, py::object cell_type,
                          cbt::RowSet const& row_set,
                          cbt::Filter const& versions,
                          std::optional<py::object> const& default_value, int /*num_workers*/,
                          int /*worker_id*/)
      : column_map_(CreateColumnMap(columns)),
        default_value_(std::move(default_value)),
        cell_type_(
            torch::python::detail::py_object_to_dtype(std::move(cell_type))),
        reader_(CreateTable(data_client, table_id, app_profile_id)
                    ->ReadRows(row_set, cbt::Filter::Chain(
                                            CreateColumnsFilter(column_map_),
                                            versions, cbt::Filter::Latest(1)))),
        it_(this->reader_.begin()) {}

  torch::Tensor next() {
    if (it_ == reader_.end()) throw py::stop_iteration();

    torch::Tensor tensor =
        getFilledTensor(this->column_map_.size(), cell_type_, default_value_);
    auto const& row = *it_;
    for (const auto& cell : row.value().cells()) {
      std::pair<std::string, std::string> key(cell.family_name(),
                                              cell.column_qualifier());
      PutCellValueInTensor(&tensor, column_map_[key], cell_type_, cell);
    }

    it_ = std::next(it_);
    return tensor;
  }

 private:
  static std::map<std::pair<std::string, std::string>, size_t> CreateColumnMap(
      py::list const& columns) {
    std::map<std::pair<std::string, std::string>, size_t> column_map;
    size_t index = 0;
    for (const auto& column_name : columns) {
      std::pair<std::string, std::string> pair =
          ColumnNameToPair(column_name.cast<std::string>());
      column_map[pair] = index++;
    }
    return column_map;
  }

  // Mapping between column names and their indices in tensors.  We're using a
  // regular map because unordered_map cannot hash a pair by default.
  std::map<std::pair<std::string, std::string>, size_t> column_map_;
  torch::Dtype cell_type_;
  std::optional<py::object> default_value_;
  cbt::RowReader reader_;
  cbt::v1::internal::RowReaderIterator it_;
};

std::shared_ptr<cbt::DataClient> CreateDataClient(
    std::string const& project_id, std::string const& instance_id,
    py::object const& /*credentials*/, std::string const& /*endpoint*/) {
  google::cloud::Options options;
  return cbt::CreateDefaultDataClient(project_id, instance_id,
                                      cbt::ClientOptions(options));
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

std::string PrintFilter(cbt::Filter const& filter) {
  std::string res;
  google::protobuf::TextFormat::PrintToString(filter.as_proto(), &res);
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
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_row_keys", &SampleRowKeys, "get sample row_keys from BigTable",
        py::arg("data_client"), py::arg("table_id"),
        py::arg("application_profile_id") = py::none());

  m.def("write_tensor", &WriteTensor, "write tensor to BigTable",
        py::arg("client"), py::arg("table_id"),
        py::arg("app_profile_id") = py::none(), py::arg("tensor"),
        py::arg("columns"), py::arg("row_keys"));

  py::class_<BigtableDatasetIterator>(m, "Iterator")
      .def(py::init<std::shared_ptr<cbt::DataClient>, std::string,
                    std::optional<std::string>, py::list, py::list, py::object,
                    cbt::RowSet const&, cbt::Filter, std::optional<py::object>, int, int>(),
           "get BigTable ReadRows iterator", py::arg("client"),
           py::arg("table_id"), py::arg("app_profile_id") = py::none(),
           py::arg("sample_row_keys"), py::arg("columns"), py::arg("cell_type"),
           py::arg("row_set"), py::arg("versions"), py::arg("default_value") = py::none(),
           py::arg("num_workers"), py::arg("worker_id"))
      .def("__iter__",
           [](BigtableDatasetIterator& it) -> BigtableDatasetIterator& {
             return it;
           })
      .def("__next__", &BigtableDatasetIterator::next);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<cbt::DataClient, std::shared_ptr<cbt::DataClient>>(
      m, "BigtableDataClient");

  m.def("create_data_client", &CreateDataClient, "Create a cbt::DataClient",
        py::arg("project_id"), py::arg("instance_id"), py::arg("credentials"),
        py::arg("endpoint"));

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
      .def("append", &AppendRowOrRange)
      .def("intersect", &cbt::RowSet::Intersect, py::arg("row_range"))
      .def("__repr__", &PrintRowSet);

  py::class_<cbt::Filter>(m, "Filter").def("__repr__", &PrintFilter);

  m.def("latest_version_filter", &cbt::Filter::Latest, py::arg("n"));
  m.def("timestamp_range_micros", &cbt::Filter::TimestampRangeMicros,
        py::arg("start"), py::arg("end"));
}
