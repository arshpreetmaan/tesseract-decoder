// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <utility>

#include "common.h"
#include "gari_two_stage_tesseract.h"
#include "stim.h"
#include "tesseract.h"
#include "utils.h"

namespace {

GariTwoStageLayout parse_gari_two_stage_layout(const nlohmann::json& root,
                                               const std::string& dem_path) {
  const auto& layout = root.at("gari_two_stage");
  if (layout.at("schema") != "gari_two_stage_layout" || layout.at("version") != 1) {
    throw std::invalid_argument("Unsupported GARI two-stage layout schema");
  }
  if (layout.at("top_prior_policy") != "modeN" ||
      layout.at("bottom_prior_policy") != "original" ||
      layout.at("dem_file") != std::filesystem::path(dem_path).filename().string()) {
    throw std::invalid_argument(
        "GARI two-stage decoding requires the mapped physical-logical mode-N DEM");
  }

  GariTwoStageLayout result;
  const auto& detectors = layout.at("detectors");
  const auto& physical_detectors = detectors.at("physical");
  const auto& virtual_detectors = detectors.at("virtual");
  result.physical_detector_count = physical_detectors.at("count").get<size_t>();
  result.virtual_detector_count = virtual_detectors.at("count").get<size_t>();
  if (physical_detectors.at("offset") != 0 ||
      virtual_detectors.at("offset") != result.physical_detector_count ||
      detectors.at("total_count") !=
          result.physical_detector_count + result.virtual_detector_count) {
    throw std::invalid_argument("GARI detector blocks are not contiguous");
  }

  const auto& errors = layout.at("errors");
  const auto& blocks = errors.at("blocks");
  size_t error_offset = 0;
  auto read_block = [&](const char* name) {
    const auto& block = blocks.at(name);
    if (block.at("offset") != error_offset) {
      throw std::invalid_argument("GARI error blocks are not contiguous");
    }
    size_t count = block.at("count").get<size_t>();
    error_offset += count;
    return count;
  };
  size_t e_z_count = read_block("e_z");
  size_t e_x_count = read_block("e_x");
  size_t e_y_count = read_block("e_y");
  size_t bar_e_z_count = read_block("bar_e_z");
  size_t bar_e_x_count = read_block("bar_e_x");
  result.physical_error_count = e_z_count + e_x_count + e_y_count;
  result.barred_error_count = bar_e_z_count + bar_e_x_count;
  if (bar_e_z_count != e_z_count || bar_e_x_count != e_x_count ||
      errors.at("physical_count") != result.physical_error_count ||
      errors.at("barred_count") != result.barred_error_count ||
      errors.at("total_count") != error_offset) {
    throw std::invalid_argument("GARI error counts do not agree with the block layout");
  }

  const auto& mappings = layout.at("barred_error_to_virtual_detector");
  if (mappings.size() != result.barred_error_count) {
    throw std::invalid_argument("GARI barred-error mapping has the wrong size");
  }
  result.barred_error_to_virtual_detector.reserve(mappings.size());
  for (size_t k = 0; k < mappings.size(); ++k) {
    if (mappings[k].at("error") != result.physical_error_count + k) {
      throw std::invalid_argument("GARI barred-error mapping is not in error order");
    }
    result.barred_error_to_virtual_detector.push_back(
        mappings[k].at("detector").get<size_t>());
  }

  if (layout.contains("top_components")) {
    const auto& top_components = layout.at("top_components");
    if (top_components.at("index_space") != "monolithic_dem") {
      throw std::invalid_argument("Unsupported GARI top-component index space");
    }
    auto read_range = [](const nlohmann::json& range) {
      return GariBlockRange{
          .offset = range.at("offset").get<size_t>(),
          .count = range.at("count").get<size_t>(),
      };
    };
    auto read_component = [&](const char* name) {
      const auto& component = top_components.at(name);
      return GariTopComponentLayout{
          .detector_rows = read_range(component.at("detector_rows")),
          .barred_error_columns = read_range(component.at("barred_error_columns")),
          .debt_detector_rows = read_range(component.at("debt_detector_rows")),
      };
    };
    result.top_components = GariTopComponentsLayout{
        .d_x = read_component("d_x"),
        .d_z = read_component("d_z"),
    };
    if (result.top_components->d_x.barred_error_columns.count != bar_e_z_count ||
        result.top_components->d_z.barred_error_columns.count != bar_e_x_count) {
      throw std::invalid_argument(
          "GARI D_X/D_Z metadata disagrees with the barred-error blocks");
    }
  }
  return result;
}

std::vector<std::vector<size_t>> map_source_orders_to_top(
    const std::vector<std::vector<size_t>>& source_orders,
    const std::vector<uint64_t>& source_to_top) {
  std::vector<std::vector<size_t>> result;
  result.reserve(source_orders.size());
  for (const auto& source_order : source_orders) {
    if (source_order.size() != source_to_top.size()) {
      throw std::invalid_argument("Source detector order has the wrong size");
    }
    std::vector<bool> seen(source_to_top.size());
    std::vector<size_t> top_order;
    top_order.reserve(source_order.size());
    for (size_t source_detector : source_order) {
      if (source_detector >= source_to_top.size() || seen[source_detector]) {
        throw std::invalid_argument("Source detector order must be a permutation");
      }
      seen[source_detector] = true;
      top_order.push_back(source_to_top[source_detector]);
    }
    result.push_back(std::move(top_order));
  }
  return result;
}

std::vector<std::vector<size_t>> build_mapped_index_orders(
    const std::vector<uint64_t>& source_to_real, size_t num_orders, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> reverse_order(0, 1);
  std::vector<std::vector<size_t>> orders(num_orders);
  for (auto& order : orders) {
    order.assign(source_to_real.begin(), source_to_real.end());
    if (reverse_order(rng)) {
      std::reverse(order.begin(), order.end());
    }
  }
  return orders;
}

std::vector<std::vector<size_t>> append_natural_virtual_suffix(
    std::vector<std::vector<size_t>> real_orders, size_t real_detectors,
    size_t virtual_detectors) {
  for (auto& order : real_orders) {
    if (order.size() != real_detectors) {
      throw std::invalid_argument("GARI real detector order has the wrong size");
    }
    std::vector<bool> seen(real_detectors);
    for (size_t detector : order) {
      if (detector >= real_detectors || seen[detector]) {
        throw std::invalid_argument("GARI real detector order must be a permutation");
      }
      seen[detector] = true;
    }
    order.reserve(real_detectors + virtual_detectors);
    for (size_t detector = 0; detector < virtual_detectors; ++detector) {
      order.push_back(real_detectors + detector);
    }
  }
  return real_orders;
}

std::vector<std::vector<size_t>> filter_custom_orders_to_top(
    const std::vector<std::vector<size_t>>& gari_orders, size_t physical_detectors,
    size_t total_detectors) {
  std::vector<std::vector<size_t>> result;
  for (const auto& gari_order : gari_orders) {
    if (gari_order.size() != total_detectors) {
      throw std::invalid_argument("Custom GARI detector order has the wrong size");
    }
    std::vector<bool> seen(total_detectors);
    std::vector<size_t> top_order;
    top_order.reserve(physical_detectors);
    for (size_t detector : gari_order) {
      if (detector >= total_detectors || seen[detector]) {
        throw std::invalid_argument("Custom GARI detector order must be a permutation");
      }
      seen[detector] = true;
      if (detector < physical_detectors) {
        top_order.push_back(detector);
      }
    }
    if (top_order.size() != physical_detectors) {
      throw std::invalid_argument("Custom GARI detector order omits a top detector");
    }
    if (std::find(result.begin(), result.end(), top_order) == result.end()) {
      result.push_back(std::move(top_order));
    }
  }
  return result;
}

}  // namespace

struct Args {
  std::string circuit_path;
  std::string dem_path;
  bool no_merge_errors = false;
  std::string det_mapping_file;
  std::string custom_order;
  bool gari_two_stage = false;
  bool gari_monolithic_one_way = false;
  bool gari_split_top = false;
  std::string gari_bottom_decoder = "tesseract";
  size_t gari_bottom_beam = 2;
  size_t gari_bottom_num_detector_orders = 1;
  size_t gari_top_candidates = 1;
  GariTwoStageLayout gari_two_stage_layout;
  std::vector<size_t> gari_source_to_top_detector;
  std::vector<std::vector<size_t>> gari_top_detector_orders;

  // Manifold orientation options
  uint64_t det_order_seed;
  size_t num_det_orders = 10;
  bool det_order_bfs = false;
  bool det_order_index = false;
  bool det_order_coordinate = false;

  // Sampling options
  size_t sample_num_shots = 0;
  size_t max_errors = SIZE_MAX;
  uint64_t sample_seed;

  // If either of these are nonzero, only the shots in the range
  // [shot_range_begin, shot_range_end) will be decoded.
  size_t shot_range_begin = 0;
  size_t shot_range_end = 0;

  // Shot data file options
  std::string in_fname = "";
  std::string in_format = "";
  std::string obs_in_fname = "";
  std::string obs_in_format = "";
  bool append_observables = false;
  std::string out_fname = "";
  std::string out_format = "";

  // If dem_out is present, a usage-frequency dem will be computed and output to
  // this file.
  std::string dem_out_fname = "";

  // If stats_out_fname is present, basic statistics and metadata will be
  // written to this file.
  std::string stats_out_fname = "";

  // The most effective way of parallelizing is over shots, confining each ILP
  // solver to a single thread.
  size_t num_threads = 1;

  // Parameters that limit the algorithm's runtime at a potential accuracy or
  // completion cost.
  size_t det_beam;
  double det_penalty = 0;
  bool beam_climbing = false;
  bool no_revisit_dets = false;

  size_t pqlimit;

  bool verbose = false;
  bool print_stats = false;

  bool sparsify_errors = false;
  int sparsify_base_degree = -1;
  int sparsify_max_degree = -1;
  int sparsify_reactivate_limit = -1;

  bool has_observables() {
    return append_observables || !obs_in_fname.empty() || (sample_num_shots > 0);
  }

  DetOrder detector_order_method() const {
    if (det_order_bfs) return DetOrder::DetBFS;
    if (det_order_coordinate) return DetOrder::DetCoordinate;
    return DetOrder::DetIndex;
  }

  std::string detector_order_method_name() const {
    if (!custom_order.empty()) return "custom";
    if (det_order_bfs) return "bfs";
    if (det_order_coordinate) return "coordinate";
    return "index";
  }

  GariBottomBackend gari_bottom_backend() const {
    return gari_bottom_decoder == "pymatching" ? GariBottomBackend::PyMatching
                                                : GariBottomBackend::Tesseract;
  }

  void validate(const argparse::ArgumentParser& program) {
    if (circuit_path.empty() and dem_path.empty()) {
      throw std::invalid_argument("Must provide at least one of --circuit or --dem");
    }

    int det_order_flags = int(det_order_bfs) + int(det_order_index) + int(det_order_coordinate);
    if (det_order_flags > 1) {
      throw std::invalid_argument(
          "Only one of --det-order-bfs, --det-order-index, or --det-order-coordinate may be set.");
    }

    int num_data_sources = int(sample_num_shots > 0) + int(!in_fname.empty());
    if (num_data_sources != 1) {
      throw std::invalid_argument("Requires exactly 1 source of shots.");
    }
    if (!in_fname.empty() and in_format.empty()) {
      throw std::invalid_argument("If --in is provided, must also specify --in-format.");
    }
    if (!out_fname.empty() and out_format.empty()) {
      throw std::invalid_argument("If --out is provided, must also specify --out-format.");
    }
    if (!in_format.empty() && !stim::format_name_to_enum_map().contains(in_format)) {
      throw std::invalid_argument("Invalid format: " + in_format);
    }
    if (!obs_in_format.empty() && !stim::format_name_to_enum_map().contains(obs_in_format)) {
      throw std::invalid_argument("Invalid format: " + obs_in_format);
    }
    if (!out_format.empty() && !stim::format_name_to_enum_map().contains(out_format)) {
      throw std::invalid_argument("Invalid format: " + out_format);
    }
    if (!obs_in_fname.empty() and in_fname.empty()) {
      throw std::invalid_argument(
          "Cannot load observable flips without a corresponding detection "
          "event data file.");
    }
    if (num_threads == 0) {
      throw std::invalid_argument("--threads must be at least 1.");
    }
    if (num_threads > 1000) {
      throw std::invalid_argument(
          "There is a maximum limit of 1000 threads imposed to avoid "
          "accidentally overloading a "
          "host. You specified " +
          std::to_string(num_threads) + "threads.");
    }
    if (shot_range_begin or shot_range_end) {
      if (shot_range_end < shot_range_begin) {
        throw std::invalid_argument("Provided shot range must have end >= begin.");
      }
    }
    if (sample_num_shots > 0 and circuit_path.empty()) {
      throw std::invalid_argument("Cannot sample shots without a circuit.");
    }
    if (beam_climbing && !gari_two_stage && det_beam == INF_DET_BEAM) {
      throw std::invalid_argument("Beam climbing requires a finite beam");
    }

    if (gari_two_stage && gari_monolithic_one_way) {
      throw std::invalid_argument(
          "--gari-two-stage and --gari-monolithic-one-way are mutually exclusive");
    }
    if (gari_split_top && !gari_two_stage) {
      throw std::invalid_argument("--gari-split-top requires --gari-two-stage");
    }
    if (gari_bottom_decoder != "tesseract" && gari_bottom_decoder != "pymatching") {
      throw std::invalid_argument(
          "--gari-bottom-decoder must be 'tesseract' or 'pymatching'");
    }
    if (!gari_two_stage && !gari_monolithic_one_way &&
        program.is_used("--gari-bottom-decoder")) {
      throw std::invalid_argument("--gari-bottom-decoder requires --gari-two-stage");
    }
    if (gari_two_stage) {
      if (dem_path.empty() || det_mapping_file.empty()) {
        throw std::invalid_argument(
            "--gari-two-stage requires --dem and --det-mapping-file");
      }
      if (!program.is_used("--beam")) {
        det_beam = 20;
      }
      if (det_beam >= INF_DET_BEAM || gari_bottom_beam >= INF_DET_BEAM) {
        throw std::invalid_argument(
            "GARI two-stage top and bottom beams must be below the infinity sentinel");
      }
      if (!program.is_used("--num-det-orders")) {
        num_det_orders = 21;
      }
      if (num_det_orders == 0) {
        throw std::invalid_argument("--num-det-orders must be at least 1");
      }
      if (gari_top_candidates == 0) {
        throw std::invalid_argument("--gari-top-candidates must be at least 1");
      }
      if (gari_bottom_num_detector_orders == 0 || gari_bottom_num_detector_orders > 2) {
        throw std::invalid_argument("--gari-bottom-num-det-orders must be 1 or 2");
      }
      if (gari_bottom_decoder == "pymatching") {
        if (gari_bottom_beam != 2 || gari_bottom_num_detector_orders != 1) {
          throw std::invalid_argument(
              "The PyMatching GARI bottom decoder requires the default "
              "--gari-bottom-beam 2 and --gari-bottom-num-det-orders 1");
        }
        if (!dem_out_fname.empty()) {
          throw std::invalid_argument(
              "--dem-out is unavailable with --gari-bottom-decoder pymatching because "
              "matching returns observables and cost, not physical error indices");
        }
      }
    }
    if (gari_monolithic_one_way) {
      if (dem_path.empty() || det_mapping_file.empty()) {
        throw std::invalid_argument(
            "--gari-monolithic-one-way requires --dem and --det-mapping-file");
      }
      if (num_det_orders == 0) {
        throw std::invalid_argument("--num-det-orders must be at least 1");
      }
      if (program.is_used("--gari-bottom-decoder") ||
          program.is_used("--gari-bottom-beam") ||
          program.is_used("--gari-bottom-num-det-orders") ||
          program.is_used("--gari-top-candidates")) {
        throw std::invalid_argument(
            "GARI two-stage bottom and candidate options cannot be used with "
            "--gari-monolithic-one-way");
      }
    }

    bool has_base = program.is_used("--sparsify-base-degree");
    bool has_max = program.is_used("--sparsify-max-degree");
    bool has_limit = program.is_used("--sparsify-reactivate-limit");

    if (!sparsify_errors) {
      if (has_base || has_max || has_limit) {
        throw std::invalid_argument(
            "Cannot use --sparsify-base-degree, --sparsify-max-degree, or "
            "--sparsify-reactivate-limit without --sparsify-errors");
      }
    } else {
      if (!has_base) {
        throw std::invalid_argument(
            "Must specify --sparsify-base-degree when --sparsify-errors is enabled.");
      }
      if (sparsify_base_degree <= 0) {
        throw std::invalid_argument("--sparsify-base-degree must be > 0.");
      }
      if (has_limit && sparsify_reactivate_limit < -1) {
        throw std::invalid_argument("--sparsify-reactivate-limit must be >= -1.");
      }
      if (has_max && sparsify_max_degree < -1) {
        throw std::invalid_argument("--sparsify-max-degree must be >= -1.");
      }
      if (sparsify_max_degree >= 0 && sparsify_max_degree < sparsify_base_degree) {
        throw std::invalid_argument("--sparsify-max-degree must be >= --sparsify-base-degree.");
      }
    }
  }

  void extract(TesseractConfig& config, std::vector<stim::SparseShot>& shots,
               std::unique_ptr<stim::MeasureRecordWriter>& writer) {
    std::vector<uint64_t> det_mapping;
    uint64_t num_original_detectors = 0;
    std::vector<std::vector<size_t>> custom_det_orders;
    if (!det_mapping_file.empty()) {
      std::ifstream f(det_mapping_file);
      if (!f) {
        throw std::invalid_argument("Could not open the mapping file: " + det_mapping_file);
      }
      nlohmann::json j = nlohmann::json::parse(f);
      num_original_detectors = j.at("num_original_detectors").get<uint64_t>();
      det_mapping = j.at("mapping").get<std::vector<uint64_t>>();
      if (gari_two_stage || gari_monolithic_one_way) {
        gari_two_stage_layout = parse_gari_two_stage_layout(j, dem_path);
        if (gari_split_top && !gari_two_stage_layout.top_components.has_value()) {
          throw std::invalid_argument(
              "--gari-split-top requires a mapping containing "
              "gari_two_stage.top_components; regenerate it with the current gari_dem_utils.py");
        }
        if (gari_two_stage_layout.physical_detector_count != num_original_detectors) {
          throw std::invalid_argument(
              "GARI two-stage layout disagrees with the source detector count");
        }
        if (det_mapping.size() != num_original_detectors) {
          throw std::invalid_argument("GARI source-detector mapping has the wrong size");
        }
        std::vector<bool> seen(num_original_detectors);
        for (size_t detector : det_mapping) {
          if (detector >= num_original_detectors || seen[detector]) {
            throw std::invalid_argument(
                "GARI source-detector mapping must be a permutation of physical detectors");
          }
          seen[detector] = true;
        }
        gari_source_to_top_detector.assign(det_mapping.begin(), det_mapping.end());
      }
      
      if (!custom_order.empty() && j.contains("det_orders")) {
        if (custom_order == "all") {
          for (auto& [key, value] : j.at("det_orders").items()) {
            custom_det_orders.push_back(value.get<std::vector<size_t>>());
          }
        } else {
          if (j.at("det_orders").contains(custom_order)) {
            custom_det_orders.push_back(j.at("det_orders").at(custom_order).get<std::vector<size_t>>());
          } else {
            throw std::invalid_argument("Mapping file does not contain custom order: " + custom_order);
          }
        }
      } else if (!custom_order.empty() &&
                 (gari_two_stage || gari_monolithic_one_way)) {
        throw std::invalid_argument(
            "--custom-order requires a mapping JSON containing det_orders");
      }
    } else if (!custom_order.empty()) {
      throw std::invalid_argument("--custom-order requires a --det-mapping-file containing the orders.");
    }

    // Get a circuit, if available
    stim::Circuit circuit;
    if (!circuit_path.empty()) {
      FILE* file = fopen(circuit_path.c_str(), "r");
      if (!file) {
        throw std::invalid_argument("Could not open the file: " + circuit_path);
      }
      circuit = stim::Circuit::from_file(file);
      fclose(file);
      if (!det_mapping.empty() && circuit.count_detectors() != det_mapping.size()) {
        throw std::invalid_argument(
            "Circuit detector count does not match the detector mapping");
      }
    }

    auto make_source_dem = [&]() {
      return stim::ErrorAnalyzer::circuit_to_detector_error_model(
          circuit, /*decompose_errors=*/false, /*fold_loops=*/true,
          /*allow_gauge_detectors=*/true,
          /*approximate_disjoint_errors_threshold=*/1,
          /*ignore_decomposition_failures=*/false,
          /*block_decomposition_from_introducing_remnant_edges=*/false);
    };

    // Get a DEM, preferring to use the specified one and falling back to
    // generating one from the circuit
    if (!dem_path.empty()) {
      FILE* file = fopen(dem_path.c_str(), "r");
      if (!file) {
        throw std::invalid_argument("Could not open the file: " + dem_path);
      }
      config.dem = stim::DetectorErrorModel::from_file(file);
      fclose(file);
    } else {
      assert(!circuit_path.empty());
      config.dem = make_source_dem();
    }

    config.merge_errors = !no_merge_errors;

    if (gari_monolithic_one_way) {
      validate_gari_monolithic_one_way_model(config.dem, gari_two_stage_layout);
      config.gari_monolithic_one_way = GariMonolithicOneWayConfig{
          .real_detector_count = gari_two_stage_layout.physical_detector_count,
          .physical_error_count = gari_two_stage_layout.physical_error_count,
      };
    }

    // Sample orientations of the error model to use for the det priority
    {
      if (verbose) {
        auto detector_coords = get_detector_coords(config.dem);
        for (size_t d = 0; d < detector_coords.size(); ++d) {
          std::cout << "Detector D" << d << " coordinate (";
          size_t e = std::min(3ul, detector_coords[d].size());
          for (size_t i = 0; i < e; ++i) {
            std::cout << detector_coords[d][i];
            if (i + 1 < e) std::cout << ", ";
          }
          std::cout << ")" << std::endl;
        }
      }
      if (gari_two_stage) {
        if (!custom_det_orders.empty()) {
          gari_top_detector_orders = filter_custom_orders_to_top(
              custom_det_orders, gari_two_stage_layout.physical_detector_count,
              gari_two_stage_layout.physical_detector_count +
                  gari_two_stage_layout.virtual_detector_count);
          num_det_orders = gari_top_detector_orders.size();
        } else if (detector_order_method() != DetOrder::DetIndex) {
          if (circuit_path.empty()) {
            throw std::invalid_argument(
                "GARI BFS/coordinate detector orders require --circuit");
          }
          auto source_orders = build_det_orders(make_source_dem(), num_det_orders,
                                                detector_order_method(), det_order_seed);
          gari_top_detector_orders = map_source_orders_to_top(source_orders, det_mapping);
        }
        config.det_orders.clear();
      } else if (gari_monolithic_one_way) {
        std::vector<std::vector<size_t>> real_orders;
        if (!custom_det_orders.empty()) {
          real_orders = filter_custom_orders_to_top(
              custom_det_orders, gari_two_stage_layout.physical_detector_count,
              gari_two_stage_layout.physical_detector_count +
                  gari_two_stage_layout.virtual_detector_count);
          num_det_orders = real_orders.size();
        } else if (detector_order_method() == DetOrder::DetIndex) {
          real_orders = build_mapped_index_orders(det_mapping, num_det_orders,
                                                  det_order_seed);
        } else {
          if (circuit_path.empty()) {
            throw std::invalid_argument(
                "GARI BFS/coordinate detector orders require --circuit");
          }
          auto source_orders = build_det_orders(make_source_dem(), num_det_orders,
                                                detector_order_method(), det_order_seed);
          real_orders = map_source_orders_to_top(source_orders, det_mapping);
        }
        config.det_orders = append_natural_virtual_suffix(
            std::move(real_orders), gari_two_stage_layout.physical_detector_count,
            gari_two_stage_layout.virtual_detector_count);
      } else if (!custom_det_orders.empty()) {
        for (const auto& order : custom_det_orders) {
          if (order.size() != config.dem.count_detectors()) {
            throw std::invalid_argument("Custom detector order size does not match DEM detector count.");
          }
        }
        config.det_orders = custom_det_orders;
        num_det_orders = custom_det_orders.size();
      } else {
        config.det_orders =
            build_det_orders(config.dem, num_det_orders, detector_order_method(), det_order_seed);
      }
    }

    if (sample_num_shots > 0) {
      assert(!circuit_path.empty());
      std::mt19937_64 rng(sample_seed);
      size_t num_detectors = circuit.count_detectors();
      const auto [dets, obs] =
          stim::sample_batch_detection_events<64>(circuit, sample_num_shots, rng);
      stim::simd_bit_table<64> obs_T = obs.transposed();
      shots.resize(sample_num_shots);
      for (size_t k = 0; k < sample_num_shots; k++) {
        shots[k].obs_mask = obs_T[k];
        for (size_t d = 0; d < num_detectors; d++) {
          if (dets[d][k]) {
            shots[k].hits.push_back(!det_mapping.empty() ? det_mapping[d] : d);
          }
        }
      }
    }

    if (!in_fname.empty()) {
      // Load the shots from a file
      FILE* shots_file = fopen(in_fname.c_str(), "r");
      if (!shots_file) {
        throw std::invalid_argument("Could not open the file: " + in_fname);
      }
      stim::FileFormatData shots_in_format = stim::format_name_to_enum_map().at(in_format);
      size_t num_dets = det_mapping.empty() ? config.dem.count_detectors() : num_original_detectors;
      auto reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          shots_file, shots_in_format.id, 0, num_dets,
          append_observables * config.dem.count_observables());

      // Load the shots from a file
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      while (reader->start_and_read_entire_record(sparse_shot)) {
        if (!det_mapping.empty()) {
          for (auto& hit : sparse_shot.hits) {
            hit = det_mapping[hit];
          }
        }
        shots.push_back(sparse_shot);
        sparse_shot.clear();
      }
      fclose(shots_file);
    }

    // Load observable flips, if applicable
    if (!obs_in_fname.empty()) {
      FILE* obs_file = fopen(obs_in_fname.c_str(), "r");
      if (!obs_file) {
        throw std::invalid_argument("Could not open the file: " + obs_in_fname);
      }
      stim::FileFormatData shots_obs_in_format = stim::format_name_to_enum_map().at(obs_in_format);
      auto obs_reader = stim::MeasureRecordReader<stim::MAX_BITWORD_WIDTH>::make(
          obs_file, shots_obs_in_format.id, 0, 0, config.dem.count_observables());
      stim::SparseShot sparse_shot;
      sparse_shot.clear();
      size_t num_obs_shots = 0;
      while (obs_reader->start_and_read_entire_record(sparse_shot)) {
        if (num_obs_shots >= shots.size()) {
          throw std::invalid_argument("Shot data ended before obs data.");
        }
        shots[num_obs_shots].obs_mask = sparse_shot.obs_mask;
        sparse_shot.clear();
        ++num_obs_shots;
      }
      if (num_obs_shots != shots.size()) {
        throw std::invalid_argument("Obs data ended before shot data ended.");
      }
      fclose(obs_file);
    }

    // Subselect shots, if applicable
    if (shot_range_begin or shot_range_end) {
      assert(shot_range_end >= shot_range_begin);
      if (shot_range_end > shots.size()) {
        throw std::invalid_argument("Shot range end is past end of shots array (size " +
                                    std::to_string(shots.size()) + ").");
      }
      std::vector<stim::SparseShot> shots_in_range(shots.begin() + shot_range_begin,
                                                   shots.begin() + shot_range_end);
      std::swap(shots_in_range, shots);
    }

    if (!out_fname.empty()) {
      // Create a writer instance to write the predicted obs to a file
      stim::FileFormatData predictions_out_format = stim::format_name_to_enum_map().at(out_format);
      FILE* predictions_file = stdout;
      if (out_fname != "-") {
        predictions_file = fopen(out_fname.c_str(), "w");
      }
      writer = stim::MeasureRecordWriter::make(predictions_file, predictions_out_format.id);
      writer->begin_result_type('L');
      // TODO: ensure the fclose happens after all predictions are written to
      // the writer.
    }
    config.det_beam = det_beam;
    config.det_penalty = det_penalty;
    config.beam_climbing = beam_climbing;
    config.no_revisit_dets = no_revisit_dets;

    config.pqlimit = pqlimit;
    config.verbose = verbose;

    config.sparsify_errors = sparsify_errors;
    config.sparsify_base_degree = sparsify_base_degree;
    config.sparsify_max_degree = sparsify_max_degree;
    config.sparsify_reactivate_limit = sparsify_reactivate_limit;
  }
};

int main(int argc, char* argv[]) {
  std::cout.precision(16);
  argparse::ArgumentParser program("tesseract");
  Args args;
  program.add_argument("--circuit").help("Stim circuit file path").store_into(args.circuit_path);
  program.add_argument("--dem").help("Stim dem file path").store_into(args.dem_path);
  program.add_argument("--det-mapping-file").help("JSON file containing detector mapping").default_value(std::string("")).store_into(args.det_mapping_file);
  program.add_argument("--custom-order")
      .help(
          "Specific detector order from a full mapping JSON (for example, 'order5' or 'all'); "
          "mapped GARI modes filter it to the real detectors")
      .default_value(std::string(""))
      .store_into(args.custom_order);
  program.add_argument("--gari-two-stage")
      .help("Split a GARI physical-logical mode-N DEM into top and bottom decoders")
      .flag()
      .store_into(args.gari_two_stage);
  program.add_argument("--gari-monolithic-one-way")
      .help(
          "Use one monolithic GARI search with real detectors first and prohibit "
          "barred errors at virtual pivots")
      .flag()
      .store_into(args.gari_monolithic_one_way);
  program.add_argument("--gari-split-top")
      .help("Decode the independent GARI D_X and D_Z top blocks separately")
      .flag()
      .store_into(args.gari_split_top);
  program.add_argument("--gari-bottom-decoder")
      .help("Bottom GARI decoder: tesseract (default) or pymatching")
      .metavar("NAME")
      .default_value(std::string("tesseract"))
      .store_into(args.gari_bottom_decoder);
  program.add_argument("--gari-bottom-beam")
      .help("Fixed detector beam for each Tesseract bottom GARI completion")
      .metavar("N")
      .default_value(size_t(2))
      .store_into(args.gari_bottom_beam);
  program.add_argument("--gari-bottom-num-det-orders")
      .help("Tesseract bottom index orders: 1 natural, or 2 natural and reverse")
      .metavar("N")
      .default_value(size_t(1))
      .store_into(args.gari_bottom_num_detector_orders);
  program.add_argument("--gari-top-candidates")
      .help(
          "Maximum completed candidates per top search (per block with --gari-split-top)")
      .metavar("N")
      .default_value(size_t(1))
      .store_into(args.gari_top_candidates);
  program.add_argument("--no-merge-errors")
      .help(
          "If provided, will not merge identical error mechanisms (two-stage children never merge)")
      .store_into(args.no_merge_errors);
  program.add_argument("--num-det-orders")
      .help("Number of detector orders (real-detector orders in mapped GARI modes)")
      .metavar("N")
      .default_value(size_t(1))
      .store_into(args.num_det_orders);
  program.add_argument("--det-order-bfs")
      .help(
          "Use BFS-based detector ordering (mapped from the source circuit in mapped GARI modes)")
      .flag()
      .store_into(args.det_order_bfs);
  program.add_argument("--det-order-index")
      .help(
          "Randomly choose increasing or decreasing detector index order "
          "(default if no method specified)")
      .flag()
      .store_into(args.det_order_index);
  program.add_argument("--det-order-coordinate")
      .help(
          "Random geometric detector orientation ordering (mapped from the source circuit in "
          "mapped GARI modes)")
      .flag()
      .store_into(args.det_order_coordinate);
  program.add_argument("--det-order-seed")
      .help(
          "Seed used when initializing the random detector traversal "
          "orderings.")
      .metavar("N")
      .default_value(static_cast<uint64_t>(518278944))
      .store_into(args.det_order_seed);
  program.add_argument("--sample-num-shots")
      .help(
          "If provided, will sample the requested number of shots from the "
          "Stim circuit and decode "
          "them. May end early if --max-errors errors are reached before "
          "decoding all shots.")
      .store_into(args.sample_num_shots);
  program.add_argument("--max-errors")
      .help(
          "If provided, will sample at least this many errors from the Stim "
          "circuit and decode "
          "them.")
      .store_into(args.max_errors);
  program.add_argument("--sample-seed")
      .help(
          "Seed used when initializing the random number generator for "
          "sampling shots")
      .metavar("N")
      .default_value(static_cast<uint64_t>(std::random_device()()))
      .store_into(args.sample_seed);
  program.add_argument("--shot-range-begin")
      .help(
          "Useful for processing a fragment of a file. If shot_range_begin == "
          "0 and shot_range_end "
          "== 0 (the default), then all available shots will be decoded. "
          "Otherwise, only those in "
          "the range [shot_range_begin, shot_range_end) will be decoded.")
      .default_value(size_t(0))
      .store_into(args.shot_range_begin);
  program.add_argument("--shot-range-end")
      .help(
          "Useful for processing a fragment of a file. If shot_range_begin == "
          "0 and shot_range_end "
          "== 0 (the default), then all available shots will be decoded. "
          "Otherwise, only those in "
          "the range [shot_range_begin, shot_range_end) will be decoded.")
      .default_value(size_t(0))
      .store_into(args.shot_range_end);
  program.add_argument("--in")
      .help("File to read detection events (and possibly observable flips) from")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.in_fname);
  std::string in_formats = "";
  bool first = true;
  for (const auto& [key, value] : stim::format_name_to_enum_map()) {
    if (!first) in_formats += "/";
    first = false;
    in_formats += key;
  }
  program.add_argument("--in-format", "--in_format")
      .help("Format of the file to read detection events from (" + in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.in_format);
  program.add_argument("--in-includes-appended-observables", "--in_includes_appended_observables")
      .help(
          "If present, assumes that the observable flips are appended to the "
          "end of each shot.")
      .default_value(false)
      .store_into(args.append_observables)
      .flag();
  program.add_argument("--obs_in", "--obs-in")
      .help("File to read observable flips from")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.obs_in_fname);
  program.add_argument("--obs-in-format", "--obs_in_format")
      .help("Format of the file to observable flips from (" + in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.obs_in_format);
  program.add_argument("--out")
      .help("File to write observable flip predictions to (or - for stdout)")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.out_fname);
  program.add_argument("--out-format")
      .help("Format of the file to write observable flip predictions to (" + in_formats + ")")
      .metavar(in_formats)
      .default_value(std::string(""))
      .store_into(args.out_format);
  program.add_argument("--dem-out")
      .help("File to write error-use frequency DEM to (full retained GARI candidate in two-stage mode)")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.dem_out_fname);
  program.add_argument("--stats-out")
      .help("File to write high-level statistics and metadata to")
      .metavar("filename")
      .default_value(std::string(""))
      .store_into(args.stats_out_fname);
  program.add_argument("--threads")
      .help("Number of decoder threads to use")
      .metavar("N")
      .default_value(size_t(
          std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency()))
      .store_into(args.num_threads);
  program.add_argument("--beam")
      .help(
          "Fixed beam, or maximum beam with climbing (default infinity; two-stage default 20; "
          "one-way GARI counts real detectors only)")
      .metavar("N")
      .default_value(INF_DET_BEAM)
      .store_into(args.det_beam);
  program.add_argument("--det-penalty")
      .help(
          "Penalty cost to add per activated detector in the residual "
          "syndrome (top child only in GARI two-stage mode).")
      .metavar("D")
      .default_value(0.0)
      .store_into(args.det_penalty);
  program.add_argument("--beam-climbing")
      .help(
          "Use beam climbing (controls the outer top schedule in two-stage mode and the ordinary "
          "monolithic schedule in one-way mode)")
      .flag()
      .store_into(args.beam_climbing);
  program.add_argument("--no-revisit-dets")
      .help(
          "Use no-revisit-dets heuristic (top child only in two-stage mode; full state in "
          "one-way mode)")
      .flag()
      .store_into(args.no_revisit_dets);

  program.add_argument("--pqlimit")
      .help(
          "Maximum priority-queue size (default infinity; applied to each child search in "
          "two-stage mode)")
      .metavar("N")
      .default_value(std::numeric_limits<size_t>::max())
      .store_into(args.pqlimit);
  program.add_argument("--verbose")
      .help("Increases output verbosity")
      .flag()
      .store_into(args.verbose);
  program.add_argument("--print-stats")
      .help(
          "Prints out the number of shots (and number of errors, if known) "
          "during decoding.")
      .flag()
      .store_into(args.print_stats);

  program.add_argument("--sparsify-errors")
      .help(
          "Enables per-shot sparse error activation (top child only in two-stage mode; "
          "barred columns only in one-way mode).")
      .flag()
      .store_into(args.sparsify_errors);
  program.add_argument("--sparsify-base-degree")
      .help(
          "Maximum detector degree for mandatory errors. Errors with degree <= K are always "
          "enabled for every shot.")
      .metavar("K")
      .scan<'i', int>()
      .store_into(args.sparsify_base_degree);
  program.add_argument("--sparsify-max-degree")
      .help(
          "Maximum detector degree for optional errors that may be reactivated. Errors with degree "
          "> M are never enabled.")
      .metavar("M")
      .scan<'i', int>()
      .store_into(args.sparsify_max_degree);
  program.add_argument("--sparsify-reactivate-limit")
      .help("Maximum number of optional errors to reactivate per shot. Use -1 for auto.")
      .metavar("N")
      .scan<'i', int>()
      .store_into(args.sparsify_reactivate_limit);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }
  TesseractConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  try {
    args.validate(program);
    args.extract(config, shots, writer);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  size_t num_observables = config.dem.count_observables();
  std::vector<stim::simd_bits<64>> obs_predicted(shots.size(),
                                                 stim::simd_bits<64>(num_observables));
  std::vector<double> cost_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  stim::DetectorErrorModel original_dem = config.dem.flattened();
  std::vector<std::unique_ptr<TesseractDecoder>> decoders(args.num_threads);
  std::vector<std::unique_ptr<GariTwoStageTesseractDecoder>> gari_decoders(args.num_threads);
  const bool collect_gari_stats = args.gari_two_stage && !args.stats_out_fname.empty();
  std::vector<size_t> gari_unique_debts(collect_gari_stats ? shots.size() : 0);
  std::vector<size_t> gari_bottom_cache_hits(collect_gari_stats ? shots.size() : 0);
  std::vector<double> gari_bottom_decode_time_seconds(collect_gari_stats ? shots.size() : 0);
  std::shared_ptr<const GariTwoStagePreparedModel> gari_prepared_model;
  GariTwoStageConfig gari_config;
  auto decoder_setup_start = std::chrono::high_resolution_clock::now();
  try {
    if (args.gari_two_stage) {
      gari_prepared_model =
          prepare_gari_two_stage_model(original_dem, args.gari_two_stage_layout,
                                       args.gari_split_top);
      gari_config.max_top_beam = args.det_beam;
      gari_config.top_beam_climbing = args.beam_climbing;
      gari_config.split_top = args.gari_split_top;
      gari_config.num_top_detector_orders = args.num_det_orders;
      gari_config.top_candidates_per_trial = args.gari_top_candidates;
      gari_config.top_detector_order_method = args.detector_order_method();
      gari_config.top_detector_order_seed = args.det_order_seed;
      gari_config.top_no_revisit_dets = args.no_revisit_dets;
      gari_config.top_detector_orders = args.gari_top_detector_orders;
      gari_config.bottom_backend = args.gari_bottom_backend();
      gari_config.bottom_beam = args.gari_bottom_beam;
      gari_config.num_bottom_detector_orders = args.gari_bottom_num_detector_orders;
      gari_config.pqlimit = args.pqlimit;
      gari_config.top_det_penalty = args.det_penalty;
      gari_config.top_sparsify_errors = args.sparsify_errors;
      gari_config.top_sparsify_base_degree = args.sparsify_base_degree;
      gari_config.top_sparsify_max_degree = args.sparsify_max_degree;
      gari_config.top_sparsify_reactivate_limit = args.sparsify_reactivate_limit;
      gari_config.collect_bottom_timing = collect_gari_stats;
      gari_config.verbose = args.verbose;
      gari_config.source_to_top_detector = args.gari_source_to_top_detector;
      for (auto& decoder : gari_decoders) {
        decoder =
            std::make_unique<GariTwoStageTesseractDecoder>(gari_prepared_model, gari_config);
      }
    } else {
      for (auto& decoder : decoders) {
        decoder = std::make_unique<TesseractDecoder>(config);
      }
    }
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  double decoder_setup_time_seconds =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - decoder_setup_start)
          .count();
  size_t error_use_count =
      !args.gari_two_stage || !args.dem_out_fname.empty() ? original_dem.count_errors() : 0;
  std::vector<std::vector<size_t>> error_use_per_thread(
      args.num_threads, std::vector<size_t>(error_use_count));
  if (args.gari_two_stage && args.dem_out_fname.empty()) {
    config.dem = stim::DetectorErrorModel();
    original_dem = stim::DetectorErrorModel();
  }
  bool has_obs = args.has_observables();
  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  double total_time_seconds = 0;
  size_t shot = parallel_for_shots_in_order(
      shots.size(), args.num_threads,
      [&](size_t thread_index, size_t shot_index) {
        auto& error_use = error_use_per_thread[thread_index];
        if (args.gari_two_stage) {
          const auto start_time = std::chrono::high_resolution_clock::now();
          auto result = gari_decoders[thread_index]->decode(shots[shot_index].hits);
          decoding_time_seconds[shot_index] =
              std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time)
                  .count();
          obs_predicted[shot_index].clear();
          for (int observable : result.observables) {
            obs_predicted[shot_index][observable] ^= 1;
          }
          low_confidence[shot_index] = !result.completed;
          cost_predicted[shot_index] = result.physical_cost;
          if (collect_gari_stats) {
            gari_unique_debts[shot_index] = result.unique_debts;
            gari_bottom_cache_hits[shot_index] = result.bottom_cache_hits;
            gari_bottom_decode_time_seconds[shot_index] = result.bottom_decode_time_seconds;
          }
          if ((!has_obs || shots[shot_index].obs_mask == obs_predicted[shot_index]) &&
              !error_use.empty()) {
            for (size_t error : result.physical_errors) {
              ++error_use[error];
            }
            for (size_t error : result.top_errors) {
              ++error_use[args.gari_two_stage_layout.physical_error_count + error];
            }
          }
        } else {
          auto& decoder = *decoders[thread_index];
          const auto start_time = std::chrono::high_resolution_clock::now();
          decoder.decode_to_errors(shots[shot_index].hits);
          decoding_time_seconds[shot_index] =
              std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time)
                  .count();
          obs_predicted[shot_index].clear();
          for (int observable : decoder.get_flipped_observables(decoder.predicted_errors_buffer)) {
            obs_predicted[shot_index][observable] ^= 1;
          }
          low_confidence[shot_index] = decoder.low_confidence_flag;
          cost_predicted[shot_index] =
              decoder.solution_cost_from_errors(decoder.predicted_errors_buffer);
          if (!has_obs || shots[shot_index].obs_mask == obs_predicted[shot_index]) {
            for (size_t ei : decoder.predicted_errors_buffer) {
              ++error_use[ei];
            }
          }
        }
      },
      [&](size_t shot_index) {
        if (writer) {
          writer->write_bits(obs_predicted[shot_index].u8, num_observables);
          writer->write_end();
        }
        if (low_confidence[shot_index]) {
          ++num_low_confidence;
        } else if (has_obs && obs_predicted[shot_index] != shots[shot_index].obs_mask) {
          ++num_errors;
        }
        total_time_seconds += decoding_time_seconds[shot_index];
        if (args.print_stats) {
          std::cout << "num_shots = " << (shot_index + 1)
                    << " num_low_confidence = " << num_low_confidence
                    << " num_errors = " << num_errors
                    << " total_time_seconds = " << total_time_seconds << std::endl;
          std::cout << "cost = " << cost_predicted[shot_index] << std::endl;
          std::cout.flush();
        }
        return num_errors < args.max_errors;
      });

  std::vector<size_t> error_use_totals(error_use_count);
  for (const auto& error_use : error_use_per_thread) {
    for (size_t ei = 0; ei < error_use_totals.size(); ++ei) {
      error_use_totals[ei] += error_use[ei];
    }
  }

  if (!args.dem_out_fname.empty()) {
    std::vector<size_t> counts(error_use_totals.begin(), error_use_totals.end());
    size_t num_usage_dem_shots = shot;
    if (has_obs) {
      // When we know the obs, we only count non-error shots.
      num_usage_dem_shots -= num_errors;
    }
    stim::DetectorErrorModel est_dem =
        common::dem_from_counts(original_dem, counts, num_usage_dem_shots);
    std::ofstream out(args.dem_out_fname, std::ofstream::out);
    if (!out.is_open()) {
      throw std::invalid_argument("Failed to open " + args.dem_out_fname);
    }
    out << est_dem << '\n';
  }

  int effective_sparsify_reactivate_limit = config.sparsify_reactivate_limit;
  if (args.gari_two_stage && args.sparsify_errors && !gari_decoders.empty()) {
    effective_sparsify_reactivate_limit =
        gari_decoders.front()->top_sparsify_reactivate_limit();
  } else {
    for (const auto& decoder : decoders) {
      if (decoder) {
        effective_sparsify_reactivate_limit = decoder->config.sparsify_reactivate_limit;
        break;
      }
    }
  }
  if (config.sparsify_errors && effective_sparsify_reactivate_limit == -1) {
    effective_sparsify_reactivate_limit = suggest_sparsify_reactivate_limit(
        config.dem.count_detectors(), config.sparsify_base_degree);
    effective_sparsify_reactivate_limit = std::min(
        effective_sparsify_reactivate_limit,
        static_cast<int>(std::min<uint64_t>(
            config.dem.count_errors(), static_cast<uint64_t>(std::numeric_limits<int>::max()))));
  }

  bool print_final_stats = true;
  if (!args.stats_out_fname.empty()) {
    nlohmann::json stats_json = {
        {"circuit_path", args.circuit_path},
        {"dem_path", args.dem_path},
        {"custom_order", args.custom_order},
        {"det_mapping_file", args.det_mapping_file},
        {"max_errors", args.max_errors},
        {"sample_seed", args.sample_seed},

        {"det_beam", args.det_beam},
        {"det_penalty", args.det_penalty},
        {"beam_climbing", args.beam_climbing},
        {"no_revisit_dets", args.no_revisit_dets},
        {"pqlimit", args.pqlimit},
        {"num_det_orders", args.num_det_orders},
        {"det_order_seed", args.det_order_seed},
        {"decoder_setup_time_seconds", decoder_setup_time_seconds},
        {"total_time_seconds", total_time_seconds},
        {"num_errors", num_errors},
        {"num_low_confidence", num_low_confidence},
        {"num_shots", shot},
        {"num_threads", args.num_threads},
        {"sample_num_shots", args.sample_num_shots},
        {"sparsify_errors", args.sparsify_errors},
        {"sparsify_base_degree", args.sparsify_base_degree},
        {"sparsify_max_degree", args.sparsify_max_degree},
        {"sparsify_reactivate_limit", effective_sparsify_reactivate_limit}};

    if (args.gari_two_stage) {
      size_t total_unique_debts = std::accumulate(
          gari_unique_debts.begin(), gari_unique_debts.begin() + shot, size_t{0});
      double bottom_decode_time_seconds = std::accumulate(
          gari_bottom_decode_time_seconds.begin(),
          gari_bottom_decode_time_seconds.begin() + shot, 0.0);
      stats_json["gari_two_stage"] = {
          {"bottom_decoder", args.gari_bottom_decoder},
          {"bottom_beam", args.gari_bottom_beam},
          {"bottom_num_det_orders", args.gari_bottom_num_detector_orders},
          {"split_top", args.gari_split_top},
          {"top_candidates_per_trial", args.gari_top_candidates},
          {"average_unique_top_candidates_per_shot",
           shot == 0 ? 0.0 : static_cast<double>(total_unique_debts) / shot},
          {"total_unique_debts_across_shots", total_unique_debts},
          {"bottom_cache_hits",
           std::accumulate(gari_bottom_cache_hits.begin(),
                           gari_bottom_cache_hits.begin() + shot, size_t{0})},
          {"bottom_decode_time_seconds", bottom_decode_time_seconds},
          {"bottom_decode_time_fraction_of_total",
           total_time_seconds <= 0
               ? 0.0
               : std::clamp(bottom_decode_time_seconds / total_time_seconds, 0.0, 1.0)},
      };
    }
    if (args.gari_monolithic_one_way) {
      stats_json["gari_monolithic_one_way"] = {
          {"real_detector_count", args.gari_two_stage_layout.physical_detector_count},
          {"virtual_detector_count", args.gari_two_stage_layout.virtual_detector_count},
          {"physical_error_count", args.gari_two_stage_layout.physical_error_count},
          {"barred_error_count", args.gari_two_stage_layout.barred_error_count},
          {"real_detector_order_method", args.detector_order_method_name()},
          {"virtual_detector_order", "natural"},
          {"beam_scope", "real_detectors"},
          {"final_cost_scope", "physical_errors"},
      };
    }

    if (args.stats_out_fname == "-") {
      std::cout << stats_json << std::endl;
      print_final_stats = false;
    } else {
      std::ofstream out(args.stats_out_fname, std::ofstream::out);
      out << stats_json << std::endl;
    }
  }
  if (print_final_stats) {
    std::cout << "num_shots = " << shot;
    std::cout << " num_low_confidence = " << num_low_confidence;
    if (has_obs) {
      std::cout << " num_errors = " << num_errors;
    }
    std::cout << " total_time_seconds = " << total_time_seconds;
    std::cout << std::endl;
  }
}
