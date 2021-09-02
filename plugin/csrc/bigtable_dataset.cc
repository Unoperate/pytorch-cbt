#include "google/cloud/bigtable/table.h"
#include "google/cloud/bigtable/table_admin.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <optional>

namespace py = pybind11;
namespace cbt = ::google::cloud::bigtable;

py::list io_big_table_sample_row_key(
    py::object client, std::string table_id,
    std::optional<std::string> app_profile_id)
{
    std::string project_id = client.attr("project_id").cast<std::string>();
    std::string instance_id = client.attr("instance_id").cast<std::string>();

    auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                        cbt::ClientOptions());

    cbt::Table table(bigtable_client, table_id);

    auto samples = table.SampleRows();

    py::list res;

    for (auto const &resp : samples.value())
    {
        std::string rk = resp.row_key;
        auto tuple = py::make_tuple(rk, resp.offset_bytes);
        res.append(tuple);
    }

    return res;
}

std::string valueToBytes(float v)
{
    // TODO make such method for every type and make them more efficient
    return std::to_string(v);
}

float bytesToValue(std::string s)
{
    // TODO
    return std::stof(s);
}

template <class T>
void putValueInTensor(torch::Tensor &tensor, int index, T value)
{
    T *ptr = tensor.data_ptr<T>();
    std::advance(ptr, index);
    *(ptr) = value;
}

void putCellValueInTensor(torch::Tensor &tensor, int index, torch::Dtype dtype,
                          cbt::Cell cell)
{
    switch (dtype)
    {
    case torch::kFloat32:
        putValueInTensor<float>(tensor, index, bytesToValue(cell.value()));
        break;

    default:
        throw std::runtime_error("type not implemented");
        break;
    }
}

void io_big_table_write(py::object client, std::string table_id,
                        std::optional<std::string> app_profile_id,
                        torch::Tensor tensor, py::list columns,
                        py::list row_keys)
{
    std::string project_id = client.attr("project_id").cast<std::string>();
    std::string instance_id = client.attr("instance_id").cast<std::string>();

    auto bigtable_client = cbt::CreateDefaultDataClient(project_id, instance_id,
                                                        cbt::ClientOptions());

    cbt::Table table(bigtable_client, table_id);

    float *ptr = (float *)tensor.data_ptr();

    for (int i = 0; i < tensor.size(0); i++)
    {
        std::string row_key = row_keys[i].cast<std::string>();

        for (int j = 0; j < tensor.size(1); j++)
        {
            std::string col_name_full = columns[j].cast<std::string>();
            std::string col_family = col_name_full.substr(0, col_name_full.find(":"));
            std::string col_name = col_name_full.substr(col_name_full.find(":") + 1,
                                                        col_name_full.length());
            google::cloud::Status status = table.Apply(cbt::SingleRowMutation(
                row_key, cbt::SetCell(std::move(col_family), std::move(col_name),
                                      valueToBytes(*ptr))));
            ptr++;
            if (!status.ok())
                throw std::runtime_error(status.message());
        }
    }
}

class BigtableTableIterator
{
public:
    BigtableTableIterator(size_t size) : size(size)
    {
        index = 0;
    };

    torch::Tensor next()
    {
        if (index == size)
            throw py::stop_iteration();

        torch::Tensor tensor = torch::empty(1, torch::TensorOptions().dtype(torch::kFloat32));
        auto *ptr = tensor.data_ptr<float>();
        (*ptr) = index++;

        return tensor;
    }

private:
    size_t size;
    size_t index;
};

BigtableTableIterator *io_big_table_iterator(
    py::object client, std::string table_id,
    std::optional<std::string> app_profile_id, py::list samples,
    py::list selected_columns, std::string start_key, std::string end_key,
    std::string row_key_prefix, std::string versions, int num_workers,
    int worker_id)
{

    std::map<std::string, size_t> column_map;
    size_t index = 0;
    for (auto &column_name : selected_columns)
    {
        column_map[column_name.cast<std::string>()] = index++;
    }

    return new BigtableTableIterator(10);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("io_big_table_sample_row_key", &io_big_table_sample_row_key,
          "get sample row_keys from BigTable", py::arg("client"),
          py::arg("table_id"), py::arg("application_profile_id") = py::none());

    m.def("io_big_table_write", &io_big_table_write, "write tensor to BigTable",
          py::arg("client"), py::arg("table_id"),
          py::arg("app_profile_id") = py::none(), py::arg("tensor"),
          py::arg("columns"), py::arg("row_keys"));

    m.def("io_big_table_iterator", &io_big_table_iterator,
          "get BigTable ReadRows iterator", py::arg("client"),
          py::arg("table_id"), py::arg("app_profile_id") = py::none(),
          py::arg("samples"), py::arg("selected_columns"), py::arg("start_key"),
          py::arg("end_key"), py::arg("row_key_prefix"), py::arg("versions"),
          py::arg("num_workers"), py::arg("worker_id"));

    py::class_<BigtableTableIterator>(m, "Iterator")
        .def("__iter__",
             [](BigtableTableIterator &it) -> BigtableTableIterator & {
                 return it;
             })
        .def("__next__", &BigtableTableIterator::next);
}