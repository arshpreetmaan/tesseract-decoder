#include "gari_two_stage_tesseract.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

struct VectorHash {
  size_t operator()(const std::vector<uint64_t>& values) const {
    size_t hash = values.size();
    for (uint64_t value : values) {
      hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

struct BottomOutcome {
  bool completed = false;
  double cost = std::numeric_limits<double>::infinity();
  std::vector<size_t> errors;
  std::vector<int> observables;
};

void validate_layout(const stim::DetectorErrorModel& dem, const GariTwoStageLayout& layout) {
  if (layout.physical_detector_count == 0) {
    throw std::invalid_argument("GARI two-stage layout has no physical detectors.");
  }
  if (layout.virtual_detector_count == 0) {
    throw std::invalid_argument("GARI two-stage layout has no virtual detectors.");
  }
  if (layout.barred_error_count != layout.virtual_detector_count) {
    throw std::invalid_argument(
        "GARI two-stage layout requires one barred error per virtual detector.");
  }
  if (layout.physical_error_count < layout.virtual_detector_count) {
    throw std::invalid_argument(
        "GARI two-stage layout is missing physical identity-block errors.");
  }
  if (layout.barred_error_to_virtual_detector.size() != layout.barred_error_count) {
    throw std::invalid_argument(
        "GARI barred-error mapping size does not match barred_error_count.");
  }

  const size_t detector_count =
      layout.physical_detector_count + layout.virtual_detector_count;
  const size_t error_count = layout.physical_error_count + layout.barred_error_count;
  if (dem.count_detectors() != detector_count) {
    throw std::invalid_argument("GARI detector count does not match the two-stage layout.");
  }
  if (dem.flattened().count_errors() != error_count) {
    throw std::invalid_argument("GARI error count does not match the two-stage layout.");
  }

  std::vector<bool> seen_virtual(layout.virtual_detector_count, false);
  for (size_t detector : layout.barred_error_to_virtual_detector) {
    if (detector < layout.physical_detector_count || detector >= detector_count) {
      throw std::invalid_argument(
          "GARI barred-error mapping references a non-virtual detector.");
    }
    size_t local_detector = detector - layout.physical_detector_count;
    if (seen_virtual[local_detector]) {
      throw std::invalid_argument(
          "GARI barred-error mapping is not one-to-one with virtual detectors.");
    }
    seen_virtual[local_detector] = true;
  }
}

bool reproduces_syndrome(const std::vector<std::vector<size_t>>& error_detectors,
                         const std::vector<size_t>& selected_errors,
                         const std::vector<uint64_t>& expected_detections,
                         size_t detector_count) {
  std::vector<bool> actual(detector_count, false);
  std::vector<bool> expected(detector_count, false);
  for (uint64_t detector : expected_detections) {
    if (detector >= detector_count || expected[detector]) {
      return false;
    }
    expected[detector] = true;
  }
  for (size_t error : selected_errors) {
    if (error >= error_detectors.size()) {
      return false;
    }
    for (size_t detector : error_detectors[error]) {
      actual[detector] = !actual[detector];
    }
  }
  return actual == expected;
}

TesseractConfig child_config(const stim::DetectorErrorModel& dem,
                             const GariTwoStageConfig& config) {
  TesseractConfig child;
  child.dem = dem;
  child.beam_climbing = false;
  // The bottom always permits revisits; the top overrides this from its
  // top-specific configuration after the common child settings are built.
  child.no_revisit_dets = false;
  child.verbose = config.verbose;
  child.merge_errors = false;
  child.pqlimit = config.pqlimit;
  child.det_penalty = config.det_penalty;
  child.create_visualization = false;
  child.sparsify_errors = false;
  return child;
}

std::vector<std::vector<size_t>> build_top_orders(const stim::DetectorErrorModel& dem,
                                                  const GariTwoStageConfig& config) {
  if (config.top_detector_order_method != DetOrder::DetIndex ||
      config.source_to_top_detector.empty()) {
    return build_det_orders(dem, config.num_top_detector_orders,
                            config.top_detector_order_method,
                            config.top_detector_order_seed);
  }
  const size_t detector_count = dem.count_detectors();
  if (config.source_to_top_detector.size() != detector_count) {
    throw std::invalid_argument("GARI source-to-top detector mapping has the wrong size.");
  }
  std::vector<bool> seen(detector_count);
  for (size_t detector : config.source_to_top_detector) {
    if (detector >= detector_count || seen[detector]) {
      throw std::invalid_argument(
          "GARI source-to-top detector mapping must be a permutation.");
    }
    seen[detector] = true;
  }

  std::mt19937_64 rng(config.top_detector_order_seed);
  std::uniform_int_distribution<int> reverse_order(0, 1);
  std::vector<std::vector<size_t>> orders(config.num_top_detector_orders);
  for (auto& order : orders) {
    order = config.source_to_top_detector;
    if (reverse_order(rng)) {
      std::reverse(order.begin(), order.end());
    }
  }
  return orders;
}

}  // namespace

GariTwoStageTesseractDecoder::GariTwoStageTesseractDecoder(
    const stim::DetectorErrorModel& gari_dem, GariTwoStageConfig config)
    : config_(std::move(config)) {
  validate_layout(gari_dem, config_.layout);

  const auto& layout = config_.layout;
  const size_t physical_detectors = layout.physical_detector_count;
  const size_t all_detectors = physical_detectors + layout.virtual_detector_count;
  const stim::DetectorErrorModel flat_dem = gari_dem.flattened();

  top_error_to_bottom_detector_.reserve(layout.barred_error_count);
  top_error_detectors_.reserve(layout.barred_error_count);
  bottom_error_detectors_.reserve(layout.physical_error_count);

  size_t error_index = 0;
  for (const stim::DemInstruction& instruction : flat_dem.instructions) {
    if (instruction.type != stim::DemInstructionType::DEM_ERROR) {
      if (instruction.type == stim::DemInstructionType::DEM_DETECTOR ||
          instruction.type == stim::DemInstructionType::DEM_LOGICAL_OBSERVABLE) {
        continue;
      }
      throw std::invalid_argument("Unsupported instruction in flattened GARI DEM: " +
                                  instruction.str());
    }
    if (instruction.arg_data.size() != 1 || !std::isfinite(instruction.arg_data[0]) ||
        instruction.arg_data[0] <= 0 || instruction.arg_data[0] >= 1) {
      throw std::invalid_argument(
          "GARI error probabilities must be strictly between zero and one.");
    }

    std::vector<size_t> physical_targets;
    std::vector<size_t> virtual_targets;
    std::unordered_set<size_t> seen_detectors;
    std::unordered_set<size_t> seen_observables;
    std::vector<stim::DemTarget> top_targets;
    std::vector<stim::DemTarget> bottom_targets;

    for (const stim::DemTarget& target : instruction.target_data) {
      if (target.is_relative_detector_id()) {
        const size_t detector = target.val();
        if (detector >= all_detectors || !seen_detectors.insert(detector).second) {
          throw std::invalid_argument("Invalid or repeated detector target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        if (detector < physical_detectors) {
          physical_targets.push_back(detector);
          top_targets.push_back(stim::DemTarget::relative_detector_id(detector));
        } else {
          virtual_targets.push_back(detector);
          bottom_targets.push_back(
              stim::DemTarget::relative_detector_id(detector - physical_detectors));
        }
      } else if (target.is_observable_id()) {
        const size_t observable = target.val();
        if (!seen_observables.insert(observable).second) {
          throw std::invalid_argument("Repeated observable target in GARI error " +
                                      std::to_string(error_index) + ".");
        }
        bottom_targets.push_back(target);
      } else {
        throw std::invalid_argument("Separator or unsupported target in GARI error " +
                                    std::to_string(error_index) + ".");
      }
    }

    if (error_index < layout.physical_error_count) {
      if (!physical_targets.empty()) {
        throw std::invalid_argument(
            "A physical GARI error violates the bottom [I U; I V] block.");
      }
      if (error_index < layout.virtual_detector_count) {
        if (virtual_targets.size() != 1 ||
            virtual_targets[0] != layout.barred_error_to_virtual_detector[error_index]) {
          throw std::invalid_argument(
              "A physical identity-block error does not target its virtual detector.");
        }
      } else if (virtual_targets.size() != 2) {
        throw std::invalid_argument(
            "A physical Y error must target exactly two virtual detectors.");
      }
      bottom_dem_.append_error_instruction(instruction.arg_data[0], bottom_targets,
                                           instruction.tag);
      std::vector<size_t> local_targets;
      local_targets.reserve(virtual_targets.size());
      for (size_t detector : virtual_targets) {
        local_targets.push_back(detector - physical_detectors);
      }
      bottom_error_detectors_.push_back(std::move(local_targets));
    } else {
      const size_t barred_error = error_index - layout.physical_error_count;
      const size_t expected_virtual = layout.barred_error_to_virtual_detector[barred_error];
      if (physical_targets.empty() || virtual_targets.size() != 1 ||
          virtual_targets[0] != expected_virtual || !seen_observables.empty()) {
        throw std::invalid_argument(
            "A barred GARI error violates the top block or its debt identity.");
      }
      top_dem_.append_error_instruction(instruction.arg_data[0], top_targets, instruction.tag);
      top_error_to_bottom_detector_.push_back(expected_virtual - physical_detectors);
      top_error_detectors_.push_back(std::move(physical_targets));
    }
    ++error_index;
  }

  if (error_index != layout.physical_error_count + layout.barred_error_count ||
      top_dem_.count_errors() != layout.barred_error_count ||
      bottom_dem_.count_errors() != layout.physical_error_count) {
    throw std::invalid_argument("GARI DEM split did not preserve all configured errors.");
  }
  if (top_dem_.count_detectors() != layout.physical_detector_count ||
      bottom_dem_.count_detectors() != layout.virtual_detector_count) {
    throw std::invalid_argument("GARI child detector dimensions do not match the layout.");
  }
  if (top_dem_.count_observables() != 0 ||
      bottom_dem_.count_observables() != gari_dem.count_observables()) {
    throw std::invalid_argument(
        "GARI observables must occur on physical errors in the bottom model only.");
  }

  if (config_.num_top_detector_orders == 0) {
    throw std::invalid_argument("GARI top detector-order count must be positive.");
  }
  if (config_.max_top_beam >= INF_DET_BEAM || config_.bottom_beam >= INF_DET_BEAM) {
    throw std::invalid_argument("GARI two-stage beams must be below the infinity sentinel.");
  }
  TesseractConfig top_config = child_config(top_dem_, config_);
  top_config.det_beam = config_.max_top_beam;
  top_config.no_revisit_dets = config_.top_no_revisit_dets;
  top_config.det_orders = build_top_orders(top_dem_, config_);

  TesseractConfig bottom_config = child_config(bottom_dem_, config_);
  bottom_config.det_beam = config_.bottom_beam;
  bottom_config.det_orders.resize(1);
  bottom_config.det_orders[0].resize(layout.virtual_detector_count);
  std::iota(bottom_config.det_orders[0].begin(), bottom_config.det_orders[0].end(), 0);

  top_decoder_ = std::make_unique<TesseractDecoder>(std::move(top_config));
  bottom_decoder_ = std::make_unique<TesseractDecoder>(std::move(bottom_config));
}

GariTwoStageDecodeResult GariTwoStageTesseractDecoder::decode(
    const std::vector<uint64_t>& top_detections) {
  const auto& layout = config_.layout;
  std::vector<bool> seen_detection(layout.physical_detector_count, false);
  for (uint64_t detector : top_detections) {
    if (detector >= layout.physical_detector_count) {
      throw std::invalid_argument("Top syndrome references a non-physical detector.");
    }
    if (seen_detection[detector]) {
      throw std::invalid_argument("Top syndrome contains a repeated detector.");
    }
    seen_detection[detector] = true;
  }

  GariTwoStageDecodeResult result;
  std::unordered_map<std::vector<uint64_t>, BottomOutcome, VectorHash> bottom_cache;

  const size_t top_trial_count = config_.top_beam_climbing
                                     ? std::max(config_.max_top_beam + 1,
                                                config_.num_top_detector_orders)
                                     : config_.num_top_detector_orders;
  for (size_t trial = 0; trial < top_trial_count; ++trial) {
    const size_t top_beam = config_.top_beam_climbing
                                ? trial % (config_.max_top_beam + 1)
                                : config_.max_top_beam;
    const size_t top_detector_order = trial % config_.num_top_detector_orders;

    top_decoder_->decode_to_errors(top_detections, top_detector_order, top_beam);

    if (top_decoder_->low_confidence_flag) {
      continue;
    }
    if (!reproduces_syndrome(top_error_detectors_, top_decoder_->predicted_errors_buffer,
                             top_detections, layout.physical_detector_count)) {
      throw std::runtime_error("Completed top GARI candidate does not reproduce its syndrome.");
    }

    std::vector<bool> debt_bits(layout.virtual_detector_count, false);
    for (size_t error : top_decoder_->predicted_errors_buffer) {
      if (error >= top_error_to_bottom_detector_.size()) {
        throw std::runtime_error("Top decoder returned an invalid barred-error index.");
      }
      size_t detector = top_error_to_bottom_detector_[error];
      debt_bits[detector] = !debt_bits[detector];
    }
    std::vector<uint64_t> debt;
    for (size_t detector = 0; detector < debt_bits.size(); ++detector) {
      if (debt_bits[detector]) {
        debt.push_back(detector);
      }
    }
    BottomOutcome outcome;
    auto cached = bottom_cache.find(debt);
    if (cached != bottom_cache.end()) {
      ++result.bottom_cache_hits;
      outcome = cached->second;
    } else {
      bottom_decoder_->decode_to_errors(debt, /*detector_order=*/0, config_.bottom_beam);

      outcome.completed = !bottom_decoder_->low_confidence_flag;
      if (outcome.completed) {
        outcome.errors = bottom_decoder_->predicted_errors_buffer;
        if (!reproduces_syndrome(bottom_error_detectors_, outcome.errors, debt,
                                 layout.virtual_detector_count)) {
          throw std::runtime_error(
              "Completed bottom GARI candidate does not reproduce its debt syndrome.");
        }
        outcome.cost = bottom_decoder_->cost_from_errors(outcome.errors);
        outcome.observables = bottom_decoder_->get_flipped_observables(outcome.errors);
      }
      bottom_cache.emplace(debt, outcome);
    }

    if (!outcome.completed) {
      continue;
    }

    if (outcome.cost < result.physical_cost) {
      result.completed = true;
      result.physical_cost = outcome.cost;
      result.observables = std::move(outcome.observables);
      result.top_errors = top_decoder_->predicted_errors_buffer;
      result.physical_errors = std::move(outcome.errors);
    }
  }
  result.unique_debts = bottom_cache.size();
  return result;
}
