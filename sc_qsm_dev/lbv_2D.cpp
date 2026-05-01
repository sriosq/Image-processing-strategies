#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

py::array_t<double> lbv2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> total_field,
    py::array_t<int, py::array::c_style | py::array::forcecast> mask,
    int max_iter = 5000,
    double tol = 1e-6,
    double omega = 1.7
) {
    auto f = total_field.unchecked<2>();
    auto m = mask.unchecked<2>();

    ssize_t nx = f.shape(0);
    ssize_t ny = f.shape(1);

    if (m.shape(0) != nx || m.shape(1) != ny) {
        throw std::runtime_error("total_field and mask must have same 2D shape.");
    }

    py::array_t<double> background({nx, ny});
    auto bg = background.mutable_unchecked<2>();

    // Initialize background field with total field
    for (ssize_t i = 0; i < nx; ++i) {
        for (ssize_t j = 0; j < ny; ++j) {
            bg(i, j) = f(i, j);
        }
    }

    // SOR solve Laplace(bg)=0 inside mask interior
    for (int iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;

        for (ssize_t i = 1; i < nx - 1; ++i) {
            for (ssize_t j = 1; j < ny - 1; ++j) {

                if (m(i, j) == 0) {
                    continue;
                }

                // Do not update boundary of the mask
                if (m(i-1, j) == 0 || m(i+1, j) == 0 ||
                    m(i, j-1) == 0 || m(i, j+1) == 0) {
                    continue;
                }

                double old_val = bg(i, j);

                double gs_val = 0.25 * (
                    bg(i-1, j) + bg(i+1, j) +
                    bg(i, j-1) + bg(i, j+1)
                );

                bg(i, j) = (1.0 - omega) * old_val + omega * gs_val;

                double change = std::abs(bg(i, j) - old_val);
                if (change > max_change) {
                    max_change = change;
                }
            }
        }

        if (max_change < tol) {
            break;
        }
    }

    py::array_t<double> local_field({nx, ny});
    auto lf = local_field.mutable_unchecked<2>();

    for (ssize_t i = 0; i < nx; ++i) {
        for (ssize_t j = 0; j < ny; ++j) {
            if (m(i, j) != 0) {
                lf(i, j) = f(i, j) - bg(i, j);
            } else {
                lf(i, j) = 0.0;
            }
        }
    }

    return local_field;
}

PYBIND11_MODULE(lbv2d, m) {
    m.doc() = "Simple 2D LBV-like background field removal solver";
    m.def("lbv2d", &lbv2d,
          py::arg("total_field"),
          py::arg("mask"),
          py::arg("max_iter") = 5000,
          py::arg("tol") = 1e-6,
          py::arg("omega") = 1.7);
}