#pragma once

#include "input/input.hpp"

#include <netcdf>
#include <stdexcept>
#include <string>
#include <vector>

class NetCDFWriter {
  public:
    NetCDFWriter(const std::string &filename, const Config &config, int n_save, int N, double dt)
        : file_(filename, netCDF::NcFile::replace), config_(config), n_save_(n_save), N_(N),
          dt_(dt) {
        define_dimensions();
        write_metadata();
    }

    void write_time(const std::vector<double> &time) {
        if (static_cast<int>(time.size()) != n_save_) {
            throw std::runtime_error("Time size does not match time dimension");
        }

        timeVar_.putVar(time.data());
    }

    void write_time_site_array(const std::string &name, const std::string &long_name,
                               const std::string &units, const std::vector<double> &data) {
        if (static_cast<int>(data.size()) != n_save_ * N_) {
            throw std::runtime_error("Data size does not match [time, site] dimensions");
        }

        std::vector<netCDF::NcDim> dims{timeDim_, siteDim_};

        netCDF::NcVar var = file_.addVar(name, netCDF::ncDouble, dims);

        var.putAtt("long_name", long_name);
        var.putAtt("units", units);
        var.putAtt("layout", "time, site");

        var.putVar(data.data());
    }

    void write_time_bond_array(const std::string &name, const std::string &long_name,
                               const std::string &units, const std::vector<double> &data) {
        const int N_bond = N_ - 1;

        if (static_cast<int>(data.size()) != n_save_ * N_bond) {
            throw std::runtime_error("Data size does not match [time, bond] dimensions");
        }

        std::vector<netCDF::NcDim> dims{timeDim_, bondDim_};

        netCDF::NcVar var = file_.addVar(name, netCDF::ncDouble, dims);

        var.putAtt("long_name", long_name);
        var.putAtt("units", units);
        var.putAtt("layout", "time, bond");

        var.putVar(data.data());
    }

  private:
    netCDF::NcFile file_;
    const Config &config_;

    int n_save_;
    int N_;
    double dt_;

    netCDF::NcDim timeDim_;
    netCDF::NcDim siteDim_;
    netCDF::NcDim bondDim_;

    netCDF::NcVar timeVar_;

    void define_dimensions() {
        timeDim_ = file_.addDim("time", n_save_);
        siteDim_ = file_.addDim("site", N_);
        bondDim_ = file_.addDim("bond", N_ - 1);

        timeVar_ = file_.addVar("time", netCDF::ncDouble, timeDim_);

        timeVar_.putAtt("long_name", "simulation time");
        timeVar_.putAtt("units", "simulation time");
    }

    void write_metadata() {
        file_.putAtt("integrator", "BAOAB");
        file_.putAtt("potential", config_.model.potential);

        file_.putAtt("N", netCDF::ncInt, config_.grid.N);
        file_.putAtt("N_time", netCDF::ncInt, config_.time.N);
        file_.putAtt("n_save", netCDF::ncInt, n_save_);
        file_.putAtt("save_every", netCDF::ncInt, config_.time.save_every);

        file_.putAtt("dt", netCDF::ncDouble, dt_);
        file_.putAtt("end_time", netCDF::ncDouble, config_.time.end_time);

        file_.putAtt("m", netCDF::ncDouble, config_.conventions.m);
        file_.putAtt("kB", netCDF::ncDouble, config_.conventions.kB);

        file_.putAtt("omega", netCDF::ncDouble, config_.model.omega);
        file_.putAtt("beta", netCDF::ncDouble, config_.model.beta);
        file_.putAtt("lambda", netCDF::ncDouble, config_.model.lambda);
        file_.putAtt("gamma", netCDF::ncDouble, config_.model.lambda / config_.conventions.m);
        file_.putAtt("left_bath_T", netCDF::ncDouble, config_.model.left_bath_T);

        file_.putAtt("N_ensemble", netCDF::ncInt, config_.ensemble.N);
    }
};