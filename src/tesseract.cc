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

#include "tesseract.h"

#include <algorithm>
#include <boost/functional/hash.hpp>  // For boost::hash_range
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>  // For std::hash (though not strictly necessary here, but good practice)
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

namespace {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  os << "]";
  return os;
}

int suggest_sparsify_reactivate_limit_capped(size_t num_detectors, int sparsify_base_degree,
                                             int max_limit) {
  if (sparsify_base_degree < 0) {
    throw std::invalid_argument("sparsify_base_degree must be >= 0.");
  }
  if (num_detectors == 0 || max_limit <= 0) {
    return 0;
  }
  double exponent = static_cast<double>(sparsify_base_degree) - 2.0;
  double max_result = static_cast<double>(max_limit);
  double log_result =
      exponent * std::log(4.5) - std::log(3.0) + std::log(static_cast<double>(num_detectors));
  if (!std::isfinite(log_result) || log_result >= std::log(max_result)) {
    return max_limit;
  }
  double result = std::exp(log_result);
  if (!std::isfinite(result)) {
    return max_limit;
  }
  double rounded = std::round(result);
  if (rounded >= max_result) {
    return max_limit;
  }
  return static_cast<int>(rounded);
}

};  // namespace

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    // Delegate to Boost's internal hash_value for dynamic_bitset
    // This is the correct and most efficient way.
    return boost::hash_value(bs);
  }
};
}  // namespace std

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";

  ss << "verbose=" << config.verbose << ", ";
  ss << "merge_errors=" << config.merge_errors << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "det_orders=" << config.det_orders << ", ";
  ss << "det_penalty=" << config.det_penalty << ", ";
  ss << "create_visualization=" << config.create_visualization;
  if (config.gari_monolithic_one_way.has_value()) {
    ss << ", gari_monolithic_one_way=true";
    if (config.gari_monolithic_one_way->bottom_beam < INF_DET_BEAM) {
      ss << ", gari_monolithic_bottom_beam="
         << config.gari_monolithic_one_way->bottom_beam;
    }
    if (config.gari_monolithic_one_way->continuation_factor > 0) {
      ss << ", gari_monolithic_continuation_factor="
         << config.gari_monolithic_one_way->continuation_factor;
    }
  }
  ss << ")";
  return ss.str();
}

int suggest_sparsify_reactivate_limit(size_t num_detectors, int sparsify_base_degree) {
  return suggest_sparsify_reactivate_limit_capped(num_detectors, sparsify_base_degree,
                                                  std::numeric_limits<int>::max());
}

std::string Node::str() {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "error_chain_idx=" << self.error_chain_idx << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  return get_detcost(d, detector_cost_tuples, d2e);
}

double TesseractDecoder::get_detcost(size_t d,
                                     const std::vector<DetectorCostTuple>& detector_cost_tuples,
                                     const std::vector<std::vector<int>>& active_d2e) const {
  return get_detcost(d, detector_cost_tuples, active_d2e, edets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples,
    const std::vector<std::vector<int>>& active_d2e,
    const std::vector<std::vector<int>>& cost_edets) const {
  double min_cost = INF;
  uint32_t min_det_cost_det_count = std::numeric_limits<uint32_t>::max();
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : active_d2e[d]) {
    ec = error_costs[ei];
    const size_t degree = cost_edets[ei].size();
    if (degree == 0) continue;
    if (ec.likelihood_cost * min_det_cost_det_count >=
        min_cost * degree)
      break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost;
      if (error_cost * min_det_cost_det_count < min_cost * dct.detectors_count) {
        min_cost = error_cost;
        min_det_cost_det_count = dct.detectors_count;
      }
    }
  }

  return (min_cost / min_det_cost_det_count) + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(std::move(config_)) {
  std::vector<size_t> dem_error_map(config.dem.flattened().count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);

  if (config.merge_errors) {
    std::vector<size_t> merge_map;
    config.dem = common::merge_indistinguishable_errors(config.dem, merge_map);
    common::chain_error_maps(dem_error_map, merge_map);
  }

  std::vector<size_t> nonzero_map;
  config.dem = common::remove_zero_probability_errors(config.dem, nonzero_map);
  common::chain_error_maps(dem_error_map, nonzero_map);

  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());

  beam_detector_count = config.dem.count_detectors();
  if (config.gari_monolithic_one_way.has_value()) {
    gari_monolithic_one_way_enabled = true;
    const auto& one_way = *config.gari_monolithic_one_way;
    if (one_way.real_detector_count > config.dem.count_detectors()) {
      throw std::invalid_argument(
          "GARI real_detector_count cannot exceed the DEM detector count.");
    }
    if (one_way.physical_error_count > dem_error_to_error.size()) {
      throw std::invalid_argument(
          "GARI physical_error_count cannot exceed the original DEM error count.");
    }
    if (!std::isfinite(one_way.continuation_factor) ||
        one_way.continuation_factor < 0) {
      throw std::invalid_argument(
          "GARI continuation_factor must be finite and nonnegative.");
    }
    beam_detector_count = one_way.real_detector_count;

    error_is_physical.resize(config.dem.count_errors());
    for (size_t error_index = 0; error_index < error_to_dem_error.size(); ++error_index) {
      size_t original_error_index = error_to_dem_error[error_index];
      if (original_error_index == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("A retained decoder error has no original DEM error mapping.");
      }
      error_is_physical[error_index] =
          original_error_index < one_way.physical_error_count;
    }
    for (size_t original_error_index = 0; original_error_index < dem_error_to_error.size();
         ++original_error_index) {
      size_t error_index = dem_error_to_error[original_error_index];
      if (error_index == std::numeric_limits<size_t>::max()) {
        continue;
      }
      bool original_is_physical = original_error_index < one_way.physical_error_count;
      if (error_is_physical[error_index] != original_is_physical) {
        throw std::invalid_argument(
            "GARI preprocessing merged physical and barred errors across the configured error "
            "boundary.");
      }
    }
  }

  if (config.det_orders.empty()) {
    config.det_orders.emplace_back(config.dem.count_detectors());
    std::iota(config.det_orders[0].begin(), config.det_orders[0].end(), 0);
  } else {
    for (size_t i = 0; i < config.det_orders.size(); ++i) {
      if (config.det_orders[i].size() != config.dem.count_detectors()) {
        throw std::invalid_argument(
            "Each detector order list must have a size equal to the number of detectors.");
      }
    }
  }
  if (config.gari_monolithic_one_way.has_value()) {
    const size_t real_detector_count =
        config.gari_monolithic_one_way->real_detector_count;
    for (const auto& detector_order : config.det_orders) {
      std::vector<uint8_t> real_seen(real_detector_count);
      for (size_t i = 0; i < real_detector_count; ++i) {
        size_t detector = detector_order[i];
        if (detector >= real_detector_count || real_seen[detector]) {
          throw std::invalid_argument(
              "Each GARI monolithic one-way detector order must begin with every real detector "
              "exactly once.");
        }
        real_seen[detector] = 1;
      }
      for (size_t i = real_detector_count; i < detector_order.size(); ++i) {
        if (detector_order[i] != i) {
          throw std::invalid_argument(
              "Each GARI monolithic one-way detector order must end with virtual detectors in "
              "natural order.");
        }
      }
    }
  }
  if (config.det_orders.empty()) {
    throw std::runtime_error("After initialization, detector orders list must not be empty.");
  }
  errors = get_errors_from_dem(config.dem.flattened());
  if (gari_monolithic_one_way_enabled) {
    for (size_t ei = 0; ei < errors.size(); ++ei) {
      bool touches_real = false;
      bool touches_virtual = false;
      for (int detector : errors[ei].symptom.detectors) {
        touches_real |= detector < beam_detector_count;
        touches_virtual |= detector >= beam_detector_count;
      }
      if (error_is_physical[ei]) {
        if (touches_real || !touches_virtual) {
          throw std::invalid_argument(
              "A physical GARI error must target virtual detectors only.");
        }
      } else if (!touches_real || !touches_virtual ||
                 !errors[ei].symptom.observables.empty()) {
        throw std::invalid_argument(
            "A barred GARI error must connect real and virtual detectors without observables.");
      }
    }
  }
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << "\n";
    }
    std::cout << std::flush;
  }
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();
  initialize_structures(config.dem.count_detectors());
  if (config.create_visualization) {
    auto detectors = get_detector_coords(config.dem);
    visualizer.add_detector_coords(detectors);
    visualizer.add_errors(errors);
  }
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.resize(num_detectors);
  edets.resize(num_errors);
  if (gari_monolithic_one_way_enabled) {
    beam_edets.resize(num_errors);
    top_error_indices.reserve(num_errors);
    bottom_error_indices.reserve(num_errors);
    for (size_t ei = 0; ei < num_errors; ++ei) {
      (error_is_physical[ei] ? bottom_error_indices : top_error_indices).push_back(ei);
    }
    gari_initial_cost_tuples.resize(num_errors);
    gari_detector_cost_tuples.resize(num_errors);
    gari_next_detector_cost_tuples.resize(num_errors);
    gari_detector_cost_cache.resize(num_detectors);
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      if (!gari_monolithic_one_way_enabled || d < beam_detector_count ||
          error_is_physical[ei]) {
        d2e[d].push_back(ei);
      }
      if (gari_monolithic_one_way_enabled && d < beam_detector_count) {
        beam_edets[ei].push_back(d);
      }
    }
  }

  for (size_t i = 0; i < errors.size(); ++i) {
    error_costs.push_back({errors[i].likelihood_cost,
                           errors[i].likelihood_cost / errors[i].symptom.detectors.size()});
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(),
              [this, d](size_t idx_a, size_t idx_b) {
                if (gari_monolithic_one_way_enabled && d < beam_detector_count) {
                  const double cost_a =
                      errors[idx_a].likelihood_cost / beam_edets[idx_a].size();
                  const double cost_b =
                      errors[idx_b].likelihood_cost / beam_edets[idx_b].size();
                  return cost_a < cost_b;
                }
                const double cost_a = error_costs[idx_a].min_cost;
                const double cost_b = error_costs[idx_b].min_cost;
                return cost_a < cost_b;
              });
  }

  if (gari_monolithic_one_way_enabled) {
    top_eneighbors.resize(num_errors);
    std::vector<boost::dynamic_bitset<>> top_edets_bitsets(
        num_errors, boost::dynamic_bitset<>(beam_detector_count));
    for (size_t ei = 0; ei < num_errors; ++ei) {
      for (int d : beam_edets[ei]) {
        top_edets_bitsets[ei][d] = 1;
      }
    }
    for (size_t ei = 0; ei < num_errors; ++ei) {
      if (beam_edets[ei].empty()) continue;
      boost::dynamic_bitset<> neighbor_set(beam_detector_count);
      for (int d : beam_edets[ei]) {
        for (int oei : d2e[d]) {
          neighbor_set |= top_edets_bitsets[oei];
        }
      }
      neighbor_set &= ~top_edets_bitsets[ei];
      for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
           d = neighbor_set.find_next(d)) {
        top_eneighbors[ei].push_back(d);
      }
    }
  }

  eneighbors.resize(num_errors);

  std::vector<boost::dynamic_bitset<>> edets_bitsets(num_errors,
                                                     boost::dynamic_bitset<>(num_detectors));
  for (size_t ei = 0; ei < num_errors; ++ei) {
    for (int d : edets[ei]) {
      edets_bitsets[ei][d] = 1;
    }
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    boost::dynamic_bitset<> neighbor_set(num_detectors, false);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        // Unify detectors from neighboring errors
        neighbor_set |= edets_bitsets[oei];
      }
    }
    // Remove detectors from error's own set
    neighbor_set &= ~edets_bitsets[ei];

    for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_set.find_next(d)) {
      eneighbors[ei].push_back(d);
    }
  }

  if (config.sparsify_errors) {
    if (config.sparsify_base_degree <= 0) {
      throw std::invalid_argument(
          "sparsify_base_degree must be > 0 when sparsify_errors is enabled.");
    }
    if (config.sparsify_max_degree < -1) {
      throw std::invalid_argument("sparsify_max_degree must be >= -1.");
    }
    if (config.sparsify_reactivate_limit < -1) {
      throw std::invalid_argument("sparsify_reactivate_limit must be >= -1.");
    }
    if (config.sparsify_max_degree >= 0 &&
        config.sparsify_max_degree < config.sparsify_base_degree) {
      throw std::invalid_argument("sparsify_max_degree must be >= sparsify_base_degree.");
    }

    if (config.sparsify_reactivate_limit == -1) {
      int error_count_limit = static_cast<int>(
          std::min(num_errors, static_cast<size_t>(std::numeric_limits<int>::max())));
      const size_t sparsify_detector_count =
          gari_monolithic_one_way_enabled ? beam_detector_count : config.dem.count_detectors();
      config.sparsify_reactivate_limit = suggest_sparsify_reactivate_limit_capped(
          sparsify_detector_count, config.sparsify_base_degree, error_count_limit);
    }

    sparsify_mandatory_errors.clear();
    sparsify_optional_errors.clear();
    for (size_t ei = 0; ei < num_errors; ++ei) {
      int degree =
          gari_monolithic_one_way_enabled && !error_is_physical[ei]
              ? static_cast<int>(beam_edets[ei].size())
              : static_cast<int>(errors[ei].symptom.detectors.size());
      // Bottom columns never overlap the initial real-only syndrome, so retain
      // all of them while sparsifying the barred/top portion of a one-way model.
      if ((gari_monolithic_one_way_enabled && error_is_physical[ei]) ||
          degree <= config.sparsify_base_degree) {
        sparsify_mandatory_errors.push_back(ei);
      } else if (degree > config.sparsify_base_degree &&
                 (config.sparsify_max_degree == -1 || degree <= config.sparsify_max_degree)) {
        sparsify_optional_errors.push_back(ei);
      }
    }
    sparse_error_active.assign(num_errors, 0);
    sparse_d2e.resize(num_detectors);
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  reset_gari_monolithic_one_way_stats();
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;

  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  if (config.det_orders.empty()) {
    throw std::runtime_error("Detector orders list must not be empty before decoding.");
  }

  if (config.beam_climbing) {
    int beam = 0;
    int detector_order = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.det_orders.size()));
         ++trial) {
      decode_to_errors_with_graph(detections, detector_order, beam, active_d2e);
      double local_cost = solution_cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
      beam += 1;
      detector_order += 1;
      beam %= (config.det_beam + 1);
      detector_order %= config.det_orders.size();
    }
  } else {
    for (size_t detector_order = 0; detector_order < config.det_orders.size(); ++detector_order) {
      decode_to_errors_with_graph(detections, detector_order, config.det_beam, active_d2e);
      double local_cost = solution_cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << detector_order << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order, int64_t error_chain_idx, boost::dynamic_bitset<>& detectors,
    std::vector<DetectorCostTuple>& detector_cost_tuples,
    const std::vector<std::vector<int>>& active_d2e,
    const std::vector<std::vector<int>>& state_edets,
    int64_t stop_before_error_chain_idx) const {
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != stop_before_error_chain_idx) {
    const auto& node = error_chain_arena[walker_idx];
    size_t ei = node.error_index;
    size_t min_detector = node.min_detector;

    for (int oei : active_d2e[min_detector]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : state_edets[ei]) {
      detectors[d] = !detectors[d];
    }
    walker_idx = node.parent_idx;
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order, size_t detector_beam) {
  reset_gari_monolithic_one_way_stats();
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;
  decode_to_errors_with_graph(detections, detector_order, detector_beam, active_d2e);
}

std::vector<std::vector<size_t>> TesseractDecoder::decode_to_error_candidates(
    const std::vector<uint64_t>& detections, size_t detector_order, size_t detector_beam,
    size_t max_candidates) {
  reset_gari_monolithic_one_way_stats();
  if (max_candidates == 0) {
    throw std::invalid_argument("max_candidates must be at least 1.");
  }
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;
  std::vector<std::vector<size_t>> candidates;
  decode_to_errors_with_graph(detections, detector_order, detector_beam, active_d2e,
                              max_candidates, &candidates);
  return candidates;
}

void TesseractDecoder::decode_to_errors_with_graph(
    const std::vector<uint64_t>& detections, size_t detector_order, size_t detector_beam,
    const std::vector<std::vector<int>>& active_d2e, size_t max_candidates,
    std::vector<std::vector<size_t>>* candidates) {
  if (gari_monolithic_one_way_enabled) {
    decode_gari_monolithic_one_way_with_graph(detections, detector_order, detector_beam,
                                               active_d2e, candidates);
    return;
  }

  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  error_chain_arena.clear();
  // Can technically be larger than pqlimit, but we need an initial guess on how many nodes we
  // will process from the queue. Only reserve if pqlimit is a reasonable (finite) value;
  // reserving SIZE_MAX bytes would throw std::length_error.
  if (config.pqlimit != std::numeric_limits<size_t>::max()) {
    error_chain_arena.reserve(config.pqlimit);
  }

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_detectors;

  boost::dynamic_bitset<> initial_detectors(num_detectors, false);
  std::vector<DetectorCostTuple> initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error(
          "Symptom " + std::to_string(d) +
          " references a detector >= num_detectors (= " + std::to_string(num_detectors) + ").");
    }
    initial_detectors[d] = true;
    for (int ei : active_d2e[d]) {
      ++initial_detector_cost_tuples[ei].detectors_count;
    }
  }

  double initial_cost = 0;
  for (size_t d : detections) {
    initial_cost += get_detcost(d, initial_detector_cost_tuples, active_d2e);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets = min_num_dets + detector_beam;

  boost::dynamic_bitset<> next_detectors;
  boost::dynamic_bitset<> detectors = initial_detectors;
  std::vector<DetectorCostTuple> detector_cost_tuples(num_errors);
  std::vector<DetectorCostTuple> next_detector_cost_tuples;
  std::vector<double> detector_cost_cache;
  pq.push({initial_cost, min_num_dets, 0, -1});
  size_t num_pq_pushed = 1;

  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();

    if (node.num_dets > max_num_dets) continue;
    detectors = initial_detectors;
    std::fill(detector_cost_tuples.begin(), detector_cost_tuples.end(), DetectorCostTuple{});
    flip_detectors_and_block_errors(detector_order, node.error_chain_idx, detectors,
                                    detector_cost_tuples, active_d2e, edets);

    if (node.num_dets == 0) {
      if (config.create_visualization) {
        visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
        visualizer.add_activated_detectors(detectors, num_detectors);
      }
      if (config.verbose) {
        std::cout << "activated_errors = ";
        int64_t walker_idx = node.error_chain_idx;
        while (walker_idx != -1) {
          std::cout << error_chain_arena[walker_idx].error_index << ", ";
          walker_idx = error_chain_arena[walker_idx].parent_idx;
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (detectors[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      if (candidates == nullptr) {
        predicted_errors_buffer.resize(node.depth);
        int64_t walker_idx = node.error_chain_idx;
        for (size_t i = 0; i < node.depth; ++i) {
          predicted_errors_buffer[node.depth - 1 - i] =
              error_to_dem_error[error_chain_arena[walker_idx].error_index];
          walker_idx = error_chain_arena[walker_idx].parent_idx;
        }
        return;
      }

      std::vector<size_t> completed_errors(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        completed_errors[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[walker_idx].error_index];
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      std::sort(completed_errors.begin(), completed_errors.end());
      if (std::find(candidates->begin(), candidates->end(), completed_errors) ==
          candidates->end()) {
        candidates->push_back(std::move(completed_errors));
      }
      if (candidates->size() >= max_candidates) {
        return;
      }
      min_num_dets = 0;
      max_num_dets = std::min(max_num_dets, detector_beam);
      continue;
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second) {
      continue;
    }

    if (config.create_visualization) {
      visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
      visualizer.add_activated_detectors(detectors, num_detectors);
    }
    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size() << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_dets = " << node.num_dets << " max_num_dets = " << max_num_dets
                << " cost = " << node.cost << std::endl;
      std::cout << "activated_errors = ";
      int64_t walker_idx = node.error_chain_idx;
      while (walker_idx != -1) {
        std::cout << error_chain_arena[walker_idx].error_index << ", ";
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (detectors[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + detector_beam + 1; i <= max_num_dets; ++i) {
          visited_detectors[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

    for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
         d = detectors.find_next(d)) {
      for (int ei : active_d2e[d]) {
        ++detector_cost_tuples[ei].detectors_count;
      }
    }

    if (next_detector_cost_tuples.empty()) {
      next_detector_cost_tuples.resize(num_errors);
    }
    next_detector_cost_tuples = detector_cost_tuples;
    next_detectors = detectors;

    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t d = 0; d < num_detectors; ++d) {
      if (detectors[config.det_orders[detector_order][d]]) {
        min_detector = config.det_orders[detector_order][d];
        break;
      }
    }

    size_t prev_ei = std::numeric_limits<size_t>::max();
    if (detector_cost_cache.empty()) {
      detector_cost_cache.resize(num_detectors);
    }
    std::fill(detector_cost_cache.begin(), detector_cost_cache.end(), -1);

    for (int ei : active_d2e[min_detector]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          next_detectors[d] = !next_detectors[d];
        }
        for (int d : edets[prev_ei]) {
          int fired = detectors[d] ? 1 : -1;
          for (int oei : active_d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;
          }
        }
      }
      prev_ei = ei;

      next_detector_cost_tuples[ei].error_blocked = 1;

      double next_cost = node.cost + errors[ei].likelihood_cost;
      size_t next_num_dets = node.num_dets;

      for (int d : edets[ei]) {
        next_detectors[d] = !next_detectors[d];
        int fired = next_detectors[d] ? 1 : -1;
        next_num_dets += fired;
        for (int oei : active_d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets && visited_detectors[next_num_dets].find(next_detectors) !=
                                        visited_detectors[next_num_dets].end())
        continue;

      for (int d : edets[ei]) {
        if (detectors[d]) {
          if (detector_cost_cache[d] == -1) {
            detector_cost_cache[d] = get_detcost(d, detector_cost_tuples, active_d2e);
          }
          next_cost -= detector_cost_cache[d];
        } else {
          next_cost += get_detcost(d, next_detector_cost_tuples, active_d2e);
        }
      }

      for (int od : eneighbors[ei]) {
        if (!detectors[od] || !next_detectors[od]) continue;
        if (detector_cost_cache[od] == -1) {
          detector_cost_cache[od] = get_detcost(od, detector_cost_tuples, active_d2e);
        }
        next_cost -= detector_cost_cache[od];
        next_cost += get_detcost(od, next_detector_cost_tuples, active_d2e);
      }

      if (next_cost == INF) continue;

      // Create the error chain node for this candidate.
      error_chain_arena.emplace_back();
      auto& next_node = error_chain_arena.back();
      next_node.error_index = ei;
      next_node.min_detector = min_detector;
      next_node.parent_idx = node.error_chain_idx;

      pq.push({next_cost, next_num_dets, node.depth + 1, (int64_t)(error_chain_arena.size() - 1)});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        if (config.verbose) {
          std::cout << (candidates != nullptr && !candidates->empty()
                            ? "stopping candidate search at priority-queue limit"
                            : "setting low confidence flag")
                    << std::endl;
        }
        low_confidence_flag = candidates == nullptr || candidates->empty();
        return;
      }
    }
  }

  if (!pq.empty()) {
    throw std::runtime_error("Priority queue should be empty after decoding failure.");
  }
  if (config.verbose && (candidates == nullptr || candidates->empty())) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = candidates == nullptr || candidates->empty();
}

void TesseractDecoder::decode_gari_monolithic_one_way_with_graph(
    const std::vector<uint64_t>& detections, size_t detector_order, size_t detector_beam,
    const std::vector<std::vector<int>>& active_d2e, std::vector<std::vector<size_t>>* candidates) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  error_chain_arena.clear();
  if (config.pqlimit != std::numeric_limits<size_t>::max()) {
    error_chain_arena.reserve(config.pqlimit);
  }

  for (size_t d : detections) {
    if (d >= beam_detector_count) {
      throw std::invalid_argument(
          "GARI monolithic one-way decoding accepts detections only on real detectors.");
    }
  }
  if (detections.empty()) {
    if (candidates != nullptr) {
      candidates->push_back({});
    }
    return;
  }

  const auto& one_way_config = *config.gari_monolithic_one_way;
  const bool collect_stats = one_way_config.collect_stats;
  const bool continuation_enabled = one_way_config.continuation_factor > 0;
  size_t num_pq_pushed = 0;
  bool pqlimit_hit = false;
  auto& initial_cost_tuples = gari_initial_cost_tuples;
  auto& detector_cost_tuples = gari_detector_cost_tuples;
  auto& next_detector_cost_tuples = gari_next_detector_cost_tuples;
  auto& detector_cost_cache = gari_detector_cost_cache;

  auto initial_phase_cost = [&](const boost::dynamic_bitset<>& state,
                                const std::vector<std::vector<int>>& state_edets,
                                const std::vector<size_t>& phase_error_indices) {
    for (size_t ei : phase_error_indices) {
      initial_cost_tuples[ei] = {};
    }
    for (size_t d = state.find_first(); d != boost::dynamic_bitset<>::npos;
         d = state.find_next(d)) {
      for (int ei : active_d2e[d]) {
        ++initial_cost_tuples[ei].detectors_count;
      }
    }
    double cost = 0;
    for (size_t d = state.find_first(); d != boost::dynamic_bitset<>::npos;
         d = state.find_next(d)) {
      cost += get_detcost(d, initial_cost_tuples, active_d2e, state_edets);
    }
    return cost;
  };

  auto collect_chain_errors = [&](int64_t chain_idx, int64_t stop_before_chain_idx) {
    std::vector<size_t> result;
    while (chain_idx != stop_before_chain_idx) {
      result.push_back(
          error_to_dem_error[error_chain_arena[chain_idx].error_index]);
      chain_idx = error_chain_arena[chain_idx].parent_idx;
    }
    std::reverse(result.begin(), result.end());
    return result;
  };

  auto add_saturating = [](size_t a, size_t b) {
    const size_t limit = std::numeric_limits<size_t>::max();
    return b > limit - a ? limit : a + b;
  };
  auto continuation_window_for = [&](size_t first_completion_pops) {
    const long double scaled =
        std::ceil(static_cast<long double>(one_way_config.continuation_factor) *
                  first_completion_pops);
    if (scaled >=
        static_cast<long double>(std::numeric_limits<size_t>::max())) {
      return std::numeric_limits<size_t>::max();
    }
    return std::max<size_t>(1, static_cast<size_t>(scaled));
  };

  std::vector<size_t> best_completed_errors;
  double best_physical_cost = std::numeric_limits<double>::max();
  bool completed_candidate_found = false;
  bool top_completion_seen = false;
  size_t continuation_window = 0;
  size_t continuation_deadline = 0;

  using PhaseRunner = std::function<std::optional<Node>(
      const boost::dynamic_bitset<>&, double, size_t,
      const std::vector<std::vector<int>>&,
      const std::vector<std::vector<int>>&,
      const std::vector<size_t>&, size_t, size_t, int64_t, size_t, int64_t,
      bool)>;
  PhaseRunner run_phase;
  std::function<bool(const Node&, size_t)> handle_top_completion;

  run_phase = [&](const boost::dynamic_bitset<>& initial_state, double initial_cost,
                  size_t phase_beam, const std::vector<std::vector<int>>& state_edets,
                  const std::vector<std::vector<int>>& state_neighbors,
                  const std::vector<size_t>& phase_error_indices,
                  size_t detector_order_begin, size_t detector_order_end,
                  int64_t chain_stop, size_t initial_depth,
                  int64_t initial_chain_idx, bool is_top) -> std::optional<Node> {
    if (initial_cost == INF) {
      return std::nullopt;
    }

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_detectors;
    const size_t initial_num_dets = initial_state.count();
    size_t min_num_dets = initial_num_dets;
    size_t max_num_dets = std::min(initial_state.size(), initial_num_dets + phase_beam);
    pq.push({initial_cost, initial_num_dets, initial_depth, initial_chain_idx});
    ++num_pq_pushed;
    if (num_pq_pushed > config.pqlimit) {
      pqlimit_hit = true;
      return std::nullopt;
    }

    boost::dynamic_bitset<> detectors(initial_state.size());
    boost::dynamic_bitset<> next_detectors(initial_state.size());
    std::optional<Node> last_completion;
    size_t phase_queue_pops = 0;

    while (!pq.empty()) {
      if (pqlimit_hit) {
        return last_completion;
      }
      if (is_top && top_completion_seen && continuation_enabled &&
          phase_queue_pops >= continuation_deadline) {
        return last_completion;
      }

      const Node node = pq.top();
      pq.pop();
      ++phase_queue_pops;
      if (collect_stats) {
        if (is_top) {
          ++gari_monolithic_one_way_stats.top_queue_pops;
          if (top_completion_seen) {
            ++gari_monolithic_one_way_stats.continuation_top_queue_pops;
          }
        } else {
          ++gari_monolithic_one_way_stats.bottom_queue_pops;
        }
      }
      if (node.num_dets > max_num_dets) continue;

      detectors = initial_state;
      for (size_t ei : phase_error_indices) {
        detector_cost_tuples[ei] = {};
      }
      flip_detectors_and_block_errors(detector_order, node.error_chain_idx, detectors,
                                      detector_cost_tuples, active_d2e, state_edets, chain_stop);

      if (node.num_dets == 0) {
        last_completion = node;
        if (config.create_visualization) {
          boost::dynamic_bitset<> full_detectors(num_detectors);
          for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
               d = detectors.find_next(d)) {
            full_detectors[d] = 1;
          }
          visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
          visualizer.add_activated_detectors(full_detectors, num_detectors);
        }
        if (is_top && handle_top_completion(node, phase_queue_pops)) {
          min_num_dets = 0;
          max_num_dets = std::min(max_num_dets, phase_beam);
          continue;
        }
        return node;
      }

      if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second) {
        continue;
      }

      if (config.create_visualization) {
        boost::dynamic_bitset<> full_detectors(num_detectors);
        for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
             d = detectors.find_next(d)) {
          full_detectors[d] = 1;
        }
        visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
        visualizer.add_activated_detectors(full_detectors, num_detectors);
      }
      if (config.verbose) {
        std::cout.precision(13);
        std::cout << (is_top ? "GARI top" : "GARI bottom") << ": len(pq) = " << pq.size()
                  << " num_pq_pushed = " << num_pq_pushed << " num_dets = " << node.num_dets
                  << " max_num_dets = " << max_num_dets << " cost = " << node.cost << std::endl;
      }

      if (node.num_dets < min_num_dets) {
        min_num_dets = node.num_dets;
        const size_t new_max_num_dets = std::min(initial_state.size(), min_num_dets + phase_beam);
        if (config.no_revisit_dets) {
          for (size_t i = new_max_num_dets + 1; i <= max_num_dets; ++i) {
            visited_detectors[i].clear();
          }
        }
        max_num_dets = std::min(max_num_dets, new_max_num_dets);
      }

      for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
           d = detectors.find_next(d)) {
        for (int ei : active_d2e[d]) {
          ++detector_cost_tuples[ei].detectors_count;
        }
      }
      for (size_t ei : phase_error_indices) {
        next_detector_cost_tuples[ei] = detector_cost_tuples[ei];
      }
      next_detectors = detectors;

      size_t min_detector = std::numeric_limits<size_t>::max();
      for (size_t i = detector_order_begin; i < detector_order_end; ++i) {
        const size_t d = config.det_orders[detector_order][i];
        if (d < detectors.size() && detectors[d]) {
          min_detector = d;
          break;
        }
      }
      if (min_detector == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("A nonempty GARI phase has no active detector pivot.");
      }

      size_t prev_ei = std::numeric_limits<size_t>::max();
      std::fill(detector_cost_cache.begin() + detector_order_begin,
                detector_cost_cache.begin() + detector_order_end, -1);
      for (int ei : active_d2e[min_detector]) {
        if (is_top == static_cast<bool>(error_is_physical[ei])) {
          throw std::runtime_error("A GARI phase exposed an error from the other phase.");
        }
        if (detector_cost_tuples[ei].error_blocked) continue;

        if (prev_ei != std::numeric_limits<size_t>::max()) {
          for (int d : state_edets[prev_ei]) {
            next_detectors[d] = !next_detectors[d];
            const int fired = detectors[d] ? 1 : -1;
            for (int oei : active_d2e[d]) {
              next_detector_cost_tuples[oei].detectors_count += fired;
            }
          }
        }
        prev_ei = ei;
        next_detector_cost_tuples[ei].error_blocked = 1;

        double next_cost = node.cost + errors[ei].likelihood_cost;
        size_t next_num_dets = node.num_dets;
        for (int d : state_edets[ei]) {
          next_detectors[d] = !next_detectors[d];
          const int fired = next_detectors[d] ? 1 : -1;
          next_num_dets += fired;
          for (int oei : active_d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;
          }
        }

        if (next_num_dets > max_num_dets) continue;
        if (config.no_revisit_dets && visited_detectors[next_num_dets].find(next_detectors) !=
                                          visited_detectors[next_num_dets].end()) {
          continue;
        }

        for (int d : state_edets[ei]) {
          if (detectors[d]) {
            if (detector_cost_cache[d] == -1) {
              detector_cost_cache[d] =
                  get_detcost(d, detector_cost_tuples, active_d2e, state_edets);
            }
            next_cost -= detector_cost_cache[d];
          } else {
            next_cost += get_detcost(d, next_detector_cost_tuples, active_d2e, state_edets);
          }
        }
        for (int d : state_neighbors[ei]) {
          if (!detectors[d] || !next_detectors[d]) continue;
          if (detector_cost_cache[d] == -1) {
            detector_cost_cache[d] = get_detcost(d, detector_cost_tuples, active_d2e, state_edets);
          }
          next_cost -= detector_cost_cache[d];
          next_cost += get_detcost(d, next_detector_cost_tuples, active_d2e, state_edets);
        }
        if (next_cost == INF) continue;

        error_chain_arena.push_back({static_cast<size_t>(ei), min_detector, node.error_chain_idx});
        const int64_t next_chain_idx = error_chain_arena.size() - 1;
        pq.push({next_cost, next_num_dets, node.depth + 1, next_chain_idx});
        ++num_pq_pushed;
        if (collect_stats && !is_top) {
          ++gari_monolithic_one_way_stats.bottom_children_generated;
          if (next_num_dets == node.num_dets) {
            ++gari_monolithic_one_way_stats.bottom_nonprogress_children_generated;
          }
        }
        if (num_pq_pushed > config.pqlimit) {
          pqlimit_hit = true;
          return last_completion;
        }
      }
    }
    return last_completion;
  };

  handle_top_completion = [&](const Node& top_solution, size_t top_queue_pops) {
    const bool is_first_completion = !top_completion_seen;
    if (collect_stats) {
      ++gari_monolithic_one_way_stats.top_completions_seen;
    }

    std::vector<size_t> top_errors =
        collect_chain_errors(top_solution.error_chain_idx, -1);

    // Debt is deliberately absent from every queued top state. Materialize it
    // only when the projected real residual reaches zero.
    boost::dynamic_bitset<> bottom_state(num_detectors);
    for (size_t d : detections) {
      bottom_state[d] = 1;
    }
    int64_t walker_idx = top_solution.error_chain_idx;
    while (walker_idx != -1) {
      const size_t ei = error_chain_arena[walker_idx].error_index;
      for (int d : edets[ei]) {
        bottom_state[d] = !bottom_state[d];
      }
      walker_idx = error_chain_arena[walker_idx].parent_idx;
    }
    for (size_t d = bottom_state.find_first();
         d != boost::dynamic_bitset<>::npos && d < beam_detector_count;
         d = bottom_state.find_next(d)) {
      throw std::runtime_error(
          "A completed projected GARI top state left a real residual.");
    }

    bool bottom_completed = true;
    std::vector<size_t> physical_errors;
    double physical_cost = 0;
    if (bottom_state.any()) {
      if (collect_stats) {
        ++gari_monolithic_one_way_stats.bottom_contexts_generated;
      }
      boost::dynamic_bitset<> debt(num_detectors - beam_detector_count);
      for (size_t d = beam_detector_count; d < num_detectors; ++d) {
        debt[d - beam_detector_count] = bottom_state[d];
      }

      auto cached = gari_bottom_cache.find(debt);
      if (cached != gari_bottom_cache.end()) {
        physical_errors = cached->second.physical_errors;
        physical_cost = cached->second.physical_cost;
        if (collect_stats) {
          ++gari_monolithic_one_way_stats.bottom_cache_hits;
        }
      } else {
        const double bottom_initial_cost =
            initial_phase_cost(bottom_state, edets, bottom_error_indices);
        if (bottom_initial_cost == INF) {
          bottom_completed = false;
        } else {
          // A nested bottom search may append many temporary error-chain
          // nodes. Copy its portable result, then reclaim that suffix before
          // resuming the still-live top queue.
          const size_t arena_checkpoint = error_chain_arena.size();
          const size_t bottom_pops_before =
              gari_monolithic_one_way_stats.bottom_queue_pops;
          std::optional<Node> bottom_solution = run_phase(
              bottom_state, bottom_initial_cost, one_way_config.bottom_beam,
              edets, eneighbors, bottom_error_indices, beam_detector_count,
              num_detectors, top_solution.error_chain_idx, top_solution.depth,
              top_solution.error_chain_idx, false);
          if (collect_stats &&
              gari_monolithic_one_way_stats.bottom_queue_pops >
                  bottom_pops_before) {
            ++gari_monolithic_one_way_stats.bottom_contexts_explored;
            unique_bottom_debts_explored.insert(debt);
            gari_monolithic_one_way_stats.unique_bottom_debts_explored =
                unique_bottom_debts_explored.size();
          }
          if (bottom_solution.has_value()) {
            physical_errors = collect_chain_errors(
                bottom_solution->error_chain_idx, top_solution.error_chain_idx);
            physical_cost = solution_cost_from_errors(physical_errors);
            gari_bottom_cache.emplace(
                debt, GariBottomCacheEntry{physical_errors, physical_cost});
          } else {
            bottom_completed = false;
          }
          error_chain_arena.resize(arena_checkpoint);
        }
      }
    }

    if (bottom_completed && physical_cost < best_physical_cost) {
      best_completed_errors = std::move(top_errors);
      best_completed_errors.insert(best_completed_errors.end(),
                                   physical_errors.begin(), physical_errors.end());
      best_physical_cost = physical_cost;
      completed_candidate_found = true;
    }
    const bool improved_shot =
        bottom_completed &&
        physical_cost < gari_monolithic_physical_incumbent;
    if (improved_shot) {
      gari_monolithic_physical_incumbent = physical_cost;
    }

    if (is_first_completion) {
      top_completion_seen = true;
      if (!continuation_enabled || pqlimit_hit) {
        return false;
      }
      continuation_window = continuation_window_for(top_queue_pops);
      continuation_deadline = add_saturating(top_queue_pops, continuation_window);
      return true;
    }

    if (improved_shot) {
      continuation_deadline = add_saturating(top_queue_pops, continuation_window);
      if (collect_stats) {
        ++gari_monolithic_one_way_stats.continuation_physical_improvements;
      }
    }
    return !pqlimit_hit;
  };

  boost::dynamic_bitset<> top_state(beam_detector_count);
  for (size_t d : detections) {
    top_state[d] = 1;
  }
  const double top_initial_cost =
      initial_phase_cost(top_state, beam_edets, top_error_indices);
  std::optional<Node> top_solution = run_phase(
      top_state, top_initial_cost, detector_beam, beam_edets, top_eneighbors,
      top_error_indices, 0, beam_detector_count, -1, 0, -1, true);
  if (collect_stats && pqlimit_hit) {
    ++gari_monolithic_one_way_stats.pqlimit_truncated_trials;
  }
  if (!top_solution.has_value() || !completed_candidate_found) {
    low_confidence_flag = true;
    if (config.verbose && !pqlimit_hit) {
      std::cout << (top_solution.has_value()
                        ? "GARI bottom search failed to converge within beam limit."
                        : "GARI top search failed to converge within beam limit.")
                << std::endl;
    }
    return;
  }
  if (config.verbose) {
    std::cout << "GARI decoding complete. Physical cost: "
              << best_physical_cost << " num_pq_pushed = " << num_pq_pushed
              << std::endl;
  }
  if (candidates == nullptr) {
    predicted_errors_buffer = std::move(best_completed_errors);
  } else {
    std::sort(best_completed_errors.begin(), best_completed_errors.end());
    candidates->push_back(std::move(best_completed_errors));
  }
}

void TesseractDecoder::reset_gari_monolithic_one_way_stats() {
  gari_monolithic_one_way_stats = {};
  gari_bottom_cache.clear();
  unique_bottom_debts_explored.clear();
  gari_monolithic_physical_incumbent = std::numeric_limits<double>::max();
}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  double total_cost = 0;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    total_cost += errors[error_index].likelihood_cost;
  }
  return total_cost;
}

double TesseractDecoder::solution_cost_from_errors(
    const std::vector<size_t>& predicted_errors) const {
  if (!config.gari_monolithic_one_way.has_value()) {
    return cost_from_errors(predicted_errors);
  }

  double total_cost = 0;
  const size_t physical_error_count =
      config.gari_monolithic_one_way->physical_error_count;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    if (dem_error_index < physical_error_count) {
      total_cost += errors[error_index].likelihood_cost;
    }
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) const {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all errors and compute the mask.
  // We use a set to perform an XOR-like sum.
  // If an observable is already in the set, we remove it (XORing with itself).
  // If it's not, we add it.
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    for (int obs_index : errors[error_index].symptom.observables) {
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

void TesseractDecoder::build_sparse_d2e(const std::vector<uint64_t>& detections) {
  if (sparse_d2e_valid && sparse_d2e_reactivate_limit == config.sparsify_reactivate_limit &&
      sparse_d2e_detections == detections) {
    return;
  }

  std::vector<uint8_t> shot_dets(num_detectors, 0);
  for (uint64_t d : detections) {
    if (d < num_detectors) {
      shot_dets[d] = 1;
    }
  }

  std::fill(sparse_error_active.begin(), sparse_error_active.end(), 0);

  for (int ei : sparsify_mandatory_errors) {
    sparse_error_active[ei] = 1;
  }

  struct OptionalErrorCandidate {
    int error_index;
    int overlap;
    int degree;
    double likelihood_cost;
  };

  std::vector<OptionalErrorCandidate> candidates;
  candidates.reserve(sparsify_optional_errors.size());

  for (int ei : sparsify_optional_errors) {
    int overlap = 0;
    for (int d : errors[ei].symptom.detectors) {
      if (shot_dets[d]) {
        overlap++;
      }
    }
    if (overlap > 0) {
      const int degree =
          gari_monolithic_one_way_enabled
              ? static_cast<int>(beam_edets[ei].size())
              : static_cast<int>(errors[ei].symptom.detectors.size());
      candidates.push_back({ei, overlap, degree, errors[ei].likelihood_cost});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const OptionalErrorCandidate& a, const OptionalErrorCandidate& b) {
              if (a.overlap != b.overlap) {
                return a.overlap > b.overlap;
              }
              if (a.degree != b.degree) {
                return a.degree < b.degree;
              }
              if (a.likelihood_cost != b.likelihood_cost) {
                return a.likelihood_cost < b.likelihood_cost;
              }
              return a.error_index < b.error_index;
            });

  size_t limit = std::min(static_cast<size_t>(config.sparsify_reactivate_limit), candidates.size());
  for (size_t i = 0; i < limit; ++i) {
    sparse_error_active[candidates[i].error_index] = 1;
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    sparse_d2e[d].clear();
    for (int ei : d2e[d]) {
      if (sparse_error_active[ei]) {
        sparse_d2e[d].push_back(ei);
      }
    }
  }
  sparse_d2e_detections = detections;
  sparse_d2e_reactivate_limit = config.sparsify_reactivate_limit;
  sparse_d2e_valid = true;
}
